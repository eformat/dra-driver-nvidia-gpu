/*
Copyright The Kubernetes Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/sirupsen/logrus"

	nvdevice "github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"github.com/NVIDIA/nvidia-container-toolkit/pkg/nvcdi"
	"github.com/NVIDIA/nvidia-container-toolkit/pkg/nvcdi/spec"
	transformroot "github.com/NVIDIA/nvidia-container-toolkit/pkg/nvcdi/transform/root"
	"k8s.io/klog/v2"

	utilcache "k8s.io/apimachinery/pkg/util/cache"
	cdiapi "tags.cncf.io/container-device-interface/pkg/cdi"
	cdiparser "tags.cncf.io/container-device-interface/pkg/parser"
	cdispec "tags.cncf.io/container-device-interface/specs-go"
	"go.yaml.in/yaml/v2"

	configapi "sigs.k8s.io/dra-driver-nvidia-gpu/api/nvidia.com/resource/v1beta1"
	"sigs.k8s.io/dra-driver-nvidia-gpu/internal/common"
)

const (
	cdiVendor      = "k8s." + DriverName
	cdiClaimClass  = "claim"
	defaultCDIRoot = "/var/run/cdi"
)

var nvidiaGPUDeviceNodePattern = regexp.MustCompile(`^/dev/nvidia(0|[1-9]\d?)$`)

type CDIHandler struct {
	logger            *logrus.Logger
	nvml              nvml.Interface
	nvdevice          nvdevice.Interface
	nvcdiClaim        nvcdi.Interface
	vfiocdi           *vfioCDIHandler
	driverRoot        string
	devRoot           string
	targetDriverRoot  string
	nvidiaCDIHookPath string

	specCache *utilcache.Expiring

	cdiRoot string
}

func NewCDIHandler(opts ...cdiOption) (*CDIHandler, error) {
	h := &CDIHandler{}
	for _, opt := range opts {
		opt(h)
	}

	if h.logger == nil {
		h.logger = logrus.New()
		h.logger.SetOutput(io.Discard)
	}
	if h.nvml == nil {
		h.nvml = nvml.New()
	}
	if h.cdiRoot == "" {
		h.cdiRoot = defaultCDIRoot
	}
	if h.nvdevice == nil {
		h.nvdevice = nvdevice.New(h.nvml)
	}

	if h.nvcdiClaim == nil {
		nvcdilib, err := nvcdi.New(
			nvcdi.WithDeviceLib(h.nvdevice),
			nvcdi.WithDriverRoot(h.driverRoot),
			nvcdi.WithDevRoot(h.devRoot),
			nvcdi.WithLogger(h.logger),
			nvcdi.WithNvmlLib(h.nvml),
			nvcdi.WithMode("nvml"),
			nvcdi.WithVendor(cdiVendor),
			nvcdi.WithClass(cdiClaimClass),
			nvcdi.WithNVIDIACDIHookPath(h.nvidiaCDIHookPath),
			nvcdi.WithFeatureFlags(nvcdi.FeatureDisableNvsandboxUtils),
		)
		if err != nil {
			return nil, fmt.Errorf("unable to create CDI library for claims: %w", err)
		}
		h.nvcdiClaim = nvcdilib
	}

	// The expiration time is defined upon key insert, not cache-globally.
	h.specCache = utilcache.NewExpiring()

	return h, nil
}

func (cdi *CDIHandler) GetCommonEditsCached() (*cdiapi.ContainerEdits, error) {
	key := "commonEdits"
	if v, ok := cdi.specCache.Get(key); ok {
		edits, ok := v.(*cdiapi.ContainerEdits)
		if !ok {
			return nil, fmt.Errorf("expected *cdiapi.ContainerEdits, got %T", v)
		}
		// Return a shallow copy so that cache entry consumer is less likely to
		// mutate the cache entry.
		clone := *edits
		return &clone, nil
	}

	t0 := time.Now()
	v, err := cdi.nvcdiClaim.GetCommonEdits()
	klog.V(7).Infof("t_cdi_get_common_edits %.3f s", time.Since(t0).Seconds())
	if err != nil {
		return nil, err
	}
	if err := validateCommonDeviceNodes(v); err != nil {
		return nil, err
	}

	// A driver-root-staged CDI contract (e.g. the NVML mock's nvidia.yaml,
	// which carries the mock's libnvidia-ml, nvidia-smi wrapper, /etc/nvml-mock
	// config and sysfs mounts plus the MOCK_NVML_* env) applies its edits to
	// containers only when the runtime resolves that spec — which DRA claim
	// containers never do: they receive THIS driver's per-claim spec alone.
	// When a contract file sits next to the claim specs in the CDI root, merge
	// it into the common edits so claim containers inherit the contract.
	if mockEdits, err := loadDriverRootCDIContract(cdi.cdiRoot); err != nil {
		klog.Warningf("Ignoring CDI contract merge: %v", err)
	} else if mockEdits != nil {
		klog.Infof("Merging driver-root CDI contract into claim common edits")
		v.ContainerEdits = mergeContainerEdits(v.ContainerEdits, mockEdits)
	}

	cdi.specCache.Set(key, v, time.Duration(5*time.Minute))
	// Return a shallow copy, see above.
	clone := *v
	return &clone, nil
}

// loadDriverRootCDIContract looks for a standalone CDI spec named
// nvidia.yaml in the CDI root and returns its container edits. nil (with no
// error) means no contract file is present — the default on real-driver
// nodes that only carry device specs.
func loadDriverRootCDIContract(cdiRoot string) (*cdispec.ContainerEdits, error) {
	path := filepath.Join(cdiRoot, "nvidia.yaml")
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}
	var parsed cdispec.Spec
	if err := yaml.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("parsing %s: %w", path, err)
	}
	edits := parsed.ContainerEdits
	return &edits, nil
}

// mergeContainerEdits appends b's env, mounts and device nodes onto a,
// skipping exact-duplicate device nodes (a and b often announce the same
// /dev nodes; duplicate device nodes in one spec are a CDI validation error).
//
// Hooks are deliberately NOT merged: the contract's update-ldcache hook runs
// inside workload images that may lack a working ldconfig, and a failing CDI
// hook aborts container creation outright — while the cache update itself is
// unnecessary on images whose default library search path already covers
// /usr/lib64 (ubi family). NVIDIA_VISIBLE_DEVICES is dropped for the same
// reason: the legacy "void" sentinel makes the driver's
// apply-cuda-memory-limits createRuntime hook exit 1 in DRA claim containers,
// where devices arrive via CDI device requests rather than the legacy env.
// Mounts under /sys are dropped too: binding the contract's sysfs over the
// container's /sys/devices hides the PCI layout the apply-cuda-memory-limits
// hook walks to resolve the allocated GPU, and the hook's failure aborts
// container creation. The mock NVML in the container keys on the device
// nodes and /etc/nvml-mock config, not on sysfs.
func mergeContainerEdits(a, b *cdispec.ContainerEdits) *cdispec.ContainerEdits {
	if a == nil {
		a = &cdispec.ContainerEdits{}
	}
	if b == nil {
		return a
	}
	for _, env := range b.Env {
		if strings.HasPrefix(env, "NVIDIA_VISIBLE_DEVICES=") {
			continue
		}
		a.Env = append(a.Env, env)
	}
	for _, m := range b.Mounts {
		// nvcdi's own common edits deliver the driver-root libraries and
		// binaries (its create-symlinks + update-ldcache hooks manage
		// /usr/lib64); duplicating them here collides with those hooks — a
		// mounted file at a path the symlink hook wants to create makes the
		// hook exit 1 and aborts container creation. The contract's unique
		// payload is its env and the /etc/nvml-mock configuration mount.
		if strings.HasPrefix(m.ContainerPath, "/sys/") ||
			strings.HasPrefix(m.ContainerPath, "/usr/lib64") ||
			strings.HasPrefix(m.ContainerPath, "/usr/bin/") {
			continue
		}
		a.Mounts = append(a.Mounts, m)
	}

	// Make the contract's /usr/lib64 library mount resolvable inside images
	// whose default loader search path omits it (debian-based CUDA images);
	// without this the mock's nvidia-smi/lib pair only works on ubi-family
	// images. Only set it when neither side already carries the variable.
	hasLdLibraryPath := false
	for _, env := range a.Env {
		if strings.HasPrefix(env, "LD_LIBRARY_PATH=") {
			hasLdLibraryPath = true
			break
		}
	}
	if !hasLdLibraryPath {
		a.Env = append(a.Env, "LD_LIBRARY_PATH=/usr/lib64")
	}

	seen := map[string]bool{}
	for _, dn := range a.DeviceNodes {
		seen[dn.Path+"|"+dn.HostPath] = true
	}
	for _, dn := range b.DeviceNodes {
		k := dn.Path + "|" + dn.HostPath
		if !seen[k] {
			seen[k] = true
			a.DeviceNodes = append(a.DeviceNodes, dn)
		}
	}
	return a
}

func (cdi *CDIHandler) WarmupDevSpecCache(uuids []string) {
	for _, uuid := range uuids {
		_, err := cdi.GetDeviceSpecsByUUIDCached(uuid)
		if err != nil {
			klog.Warningf("Ignore error during cache warmup: GetDeviceSpecsByUUIDCached() failed: %s", err)
		}
	}
}

func (cdi *CDIHandler) GetDeviceSpecsByUUIDCached(uuid string) ([]cdispec.Device, error) {
	key := uuid
	if v, ok := cdi.specCache.Get(key); ok {
		devs, ok := v.([]cdispec.Device)
		if !ok {
			return nil, fmt.Errorf("expected []cdispec.Device, got %T", v)
		}
		clone := make([]cdispec.Device, len(devs))
		copy(clone, devs)
		return clone, nil
	}

	t0 := time.Now()
	devs, err := cdi.nvcdiClaim.GetDeviceSpecsByID(uuid)
	klog.V(6).Infof("GetDeviceSpecsByID() called for %s, t_cdi_get_device_specs_by_id %.3f s", uuid, time.Since(t0).Seconds())
	if err != nil {
		return nil, err
	}
	if err := validateGPUDeviceNodes(uuid, devs); err != nil {
		return nil, err
	}
	cdi.stripMockCUDAHeadroomHooks(devs)
	cdi.specCache.Set(key, devs, time.Duration(5*time.Minute))
	clone := make([]cdispec.Device, len(devs))
	copy(clone, devs)
	return clone, nil
}

// InvalidateDeviceSpec removes cached CDI data for a GPU whose device minor
// may have changed after rebinding it to the NVIDIA driver.
func (cdi *CDIHandler) InvalidateDeviceSpec(uuid string) {
	cdi.specCache.Delete(uuid)
}

func (cdi *CDIHandler) getDeviceSpecsByUUIDAndMinor(uuid string, minor int) ([]cdispec.Device, error) {
	devices, err := cdi.GetDeviceSpecsByUUIDCached(uuid)
	if err != nil {
		return nil, err
	}
	if err := validateGPUDeviceMinor(uuid, minor, devices); err == nil {
		return devices, nil
	}

	// The GPU minor may have changed after a driver rebind or MIG mode change.
	cdi.InvalidateDeviceSpec(uuid)
	devices, err = cdi.GetDeviceSpecsByUUIDCached(uuid)
	if err != nil {
		return nil, err
	}
	if err := validateGPUDeviceMinor(uuid, minor, devices); err != nil {
		return nil, err
	}
	return devices, nil
}

// stripMockCUDAHeadroomHooks removes nvcdi's apply-cuda-memory-limits
// createRuntime hooks when running against the NVML mock driver tree. The hook
// executes inside the workload container and resolves libnvidia-ml there — on
// images that ship a real NVML (CUDA images) it drives the mock's character
// devices with real NVML and exits 1, aborting container creation outright.
// The hook enforces per-MIG-device CUDA memory limits, which the mock does not
// model. Real-driver roots keep the hook untouched.
func (cdi *CDIHandler) stripMockCUDAHeadroomHooks(devs []cdispec.Device) {
	if !strings.Contains(cdi.targetDriverRoot, "nvml-mock") {
		return
	}
	stripped := 0
	for i := range devs {
		edits := &devs[i].ContainerEdits
		kept := edits.Hooks[:0]
		for _, hook := range edits.Hooks {
			isCUDAHeadroom := false
			for _, arg := range hook.Args {
				if arg == "apply-cuda-memory-limits" {
					isCUDAHeadroom = true
					break
				}
			}
			if isCUDAHeadroom {
				stripped++
				continue
			}
			kept = append(kept, hook)
		}
		edits.Hooks = kept
	}
	if stripped > 0 {
		klog.V(1).Infof("Stripped %d apply-cuda-memory-limits hook(s): mock driver root, workload images carrying real NVML cannot drive the mock's device nodes", stripped)
	}
}

// validateCommonDeviceNodes verifies that nvcdi emitted the global NVIDIA
// control and UVM device nodes required by GPU and MIG workloads. nvcdi's
// character-device discoverer resolves these paths against its configured
// devRoot and omits nodes that do not exist there.
func validateCommonDeviceNodes(edits *cdiapi.ContainerEdits) error {
	requiredPaths := []string{
		"/dev/nvidiactl",
		"/dev/nvidia-uvm",
		"/dev/nvidia-uvm-tools",
	}
	foundPaths := make(map[string]struct{}, len(requiredPaths))
	if edits != nil && edits.ContainerEdits != nil {
		for _, node := range edits.DeviceNodes {
			if node != nil {
				foundPaths[node.Path] = struct{}{}
			}
		}
	}

	var missingPaths []string
	for _, path := range requiredPaths {
		if _, found := foundPaths[path]; !found {
			missingPaths = append(missingPaths, path)
		}
	}
	if len(missingPaths) > 0 {
		return fmt.Errorf("failed to validate NVIDIA CDI common edits: missing required device nodes %s; NVIDIA driver installation may be incomplete", strings.Join(missingPaths, ", "))
	}
	return nil
}

// validateGPUDeviceNodes verifies that nvcdi found a per-GPU device node.
func validateGPUDeviceNodes(uuid string, devices []cdispec.Device) error {
	for _, device := range devices {
		for _, node := range device.ContainerEdits.DeviceNodes {
			if node != nil && nvidiaGPUDeviceNodePattern.MatchString(node.Path) {
				return nil
			}
		}
	}
	return fmt.Errorf("failed to validate NVIDIA CDI device spec for GPU %q: missing /dev/nvidia<minor> device node; NVIDIA driver installation may be incomplete", uuid)
}

func validateGPUDeviceMinor(uuid string, minor int, devices []cdispec.Device) error {
	if err := validateGPUDeviceNodes(uuid, devices); err != nil {
		return err
	}

	expectedPath := fmt.Sprintf("/dev/nvidia%d", minor)
	for _, device := range devices {
		for _, node := range device.ContainerEdits.DeviceNodes {
			if node != nil && node.Path == expectedPath {
				return nil
			}
		}
	}
	return fmt.Errorf("failed to validate NVIDIA CDI device spec for GPU %q: expected device node %s", uuid, expectedPath)
}

// Note(JP): for a regular GPU, this canonical name is for example `gpu-0`, with
// the numerical suffix as of the time of writing reflecting the device minor.
// NVMLs' DeviceSetMigMode() is documented with 'This API may unbind or reset
// the device to activate the requested mode. Thus, the attributes associated
// with the device, such as minor number, might change. The caller of this API
// is expected to query such attributes again.' -- if the minor is indeed not
// necessarily stable, there may be problems associating this spec _long-term_
// with that name. That is an argument for always dynamically generating also
// full-GPU CDI spec during prepare() (or: to cache it, and re-generate it every
// now and then during this program's lifetime).
func (cdi *CDIHandler) CreateClaimSpecFile(claimUID string, preparedDevices PreparedDevices, claimContainerEdits *cdiapi.ContainerEdits) error {
	commonEdits := &cdiapi.ContainerEdits{ContainerEdits: &cdispec.ContainerEdits{}}
	if requiresNVIDIADeviceNodes(preparedDevices) {
		// Generate those parts of the container spec that are not device-specific
		// (to inject e.g. driver library mounts and meta devices). Note that
		// `nvcdiDevice.GetCommonEdits()` may usually initialize nvsandboxutilslib
		// under the hood -- we now prevent that from happening by using
		// `nvcdi.FeatureDisableNvsandboxUtils` above.
		var err error
		commonEdits, err = cdi.GetCommonEditsCached()
		if err != nil {
			return fmt.Errorf("failed to get common CDI spec edits: %w", err)
		}
	}

	var deviceSpecs []cdispec.Device

	for _, group := range preparedDevices {
		for _, dev := range group.Devices {
			uuid := ""

			// Construct claim-specific CDI device name in accordance with the
			// naming convention encoded in `GetClaimDeviceName()` below.
			dname := fmt.Sprintf("%s-%s", claimUID, dev.CanonicalName())

			var dspec cdispec.Device

			if dev.Type() == GpuDeviceType {
				uuid = dev.Gpu.Info.UUID
				// Get (copy of) cached CDI spec (is safe to be mutated below,
				// w/o compromising cache).
				dspecsgpu, err := cdi.getDeviceSpecsByUUIDAndMinor(uuid, dev.Gpu.Info.minor)
				if err != nil {
					return fmt.Errorf("unable to get device spec for %s: %w", dname, err)
				}
				dspec = dspecsgpu[0]
			}

			if dev.Type() == VfioDeviceType {
				castConfig, ok := group.ConfigState.Config.(*configapi.VfioDeviceConfig)
				if !ok {
					return fmt.Errorf("received invalid config type for the vfio device")
				}
				dspecsvfio, err := cdi.vfiocdi.GetDeviceSpecsByPCIBusID(dev.Vfio.Info.PciBusID, castConfig.Iommu.ShouldPreferIommuFD())
				if err != nil {
					return fmt.Errorf("failed to get CDI container edits for vfio device: %w", err)
				}
				dspec = dspecsvfio[0]
			}

			if dev.Type() == PreparedMigDeviceType {
				// Here, get the 'parent dev node' part of the spec. THe spec
				// fragment for other dev nodes specific to this MIG device is
				// generated further below. One reason for doing things this way
				// is that `nvcdiDevice.GetDeviceSpecsByID(MIG_UUID)` may yield
				// an incomplete spec for MIG devices, see
				// https://github.com/kubernetes-sigs/dra-driver-nvidia-gpu/issues/787.
				uuid = dev.Mig.Concrete.ParentUUID
				// Get (copy of) cached device spec (is safe to be mutated below,
				// w/o compromising cache).
				dspecsmig, err := cdi.getDeviceSpecsByUUIDAndMinor(uuid, int(dev.Mig.Concrete.ParentMinor))
				if err != nil {
					return fmt.Errorf("unable to get device spec for %s: %w", dname, err)
				}
				dspec = dspecsmig[0]

				devnodesForMig, err := cdi.GetDevNodesForMigDevice(dev.Mig.Concrete)
				if err != nil {
					return fmt.Errorf("failed to construct MIG device DeviceNode edits: %w", err)
				}
				klog.V(7).Infof("CDI spec: appending MIG device nodes")
				dspec.ContainerEdits.DeviceNodes = append(dspec.ContainerEdits.DeviceNodes, devnodesForMig...)
			}

			// Associate thew newly generated spec with the claim-specific
			// device name.
			dspec.Name = dname

			// If there are edits passed as part of the device config state (set
			// on the group), add them to the spec for this device.
			if group.ConfigState.containerEdits != nil {
				var isVfioDeviceConfigType bool
				if group.ConfigState.Config != nil {
					// If claim requests vfio devices, we assume that all vfio
					// device requests use the same config. We also assume that
					// other device types are not requested in the same claim.
					// With that, we expect to only encounter this codepath once
					// and override the common edits that's initialized with nvcdi specs.
					_, isVfioDeviceConfigType = group.ConfigState.Config.(*configapi.VfioDeviceConfig)
					if isVfioDeviceConfigType {
						commonEdits = group.ConfigState.containerEdits
					}
				}

				// Only non-vfio devices require the group config state to be applied to
				// the device cdi spec.
				if !isVfioDeviceConfigType {
					deviceEdits := &cdiapi.ContainerEdits{
						ContainerEdits: &dspec.ContainerEdits,
					}
					deviceEdits = deviceEdits.Append(group.ConfigState.containerEdits)
					dspec.ContainerEdits = *deviceEdits.ContainerEdits
				}
			}
			klog.V(7).Infof("Number of device nodes about to inject for device %s: %d", dname, len(dspec.ContainerEdits.DeviceNodes))
			deviceSpecs = append(deviceSpecs, dspec)
		}
	}

	if claimContainerEdits != nil {
		// Claim-level edits apply whenever any device from this claim-specific
		// CDI spec is injected.
		commonEdits = (&cdiapi.ContainerEdits{}).Append(commonEdits).Append(claimContainerEdits)
	}

	tws0 := time.Now()
	spec, err := spec.New(
		spec.WithVendor(cdiVendor),
		spec.WithClass(cdiClaimClass),
		spec.WithDeviceSpecs(deviceSpecs),
		spec.WithEdits(*commonEdits.ContainerEdits),
	)
	if err != nil {
		return fmt.Errorf("failed to create CDI spec: %w", err)
	}

	// Write the per-claim spec that was generated above to the filesystem. As
	// it is bound to a DRA ResourceClaim, it's transient (bound to the lifetime
	// of a container). Hence, Use the "transient spec" concept from CDI.
	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiClaimClass, claimUID)
	klog.V(6).Infof("Writing CDI spec '%s' for claim '%s'", specName, claimUID)
	result := cdi.writeSpec(spec, specName)

	klog.V(7).Infof("t_gen_write_cdi_spec %.3f s", time.Since(tws0).Seconds())
	return result
}

// requiresNVIDIADeviceNodes reports whether the claim needs NVIDIA control and
// GPU device nodes. VFIO-only claims use /dev/vfio devices instead.
func requiresNVIDIADeviceNodes(preparedDevices PreparedDevices) bool {
	for _, group := range preparedDevices {
		for _, device := range group.Devices {
			if device.Type() == GpuDeviceType || device.Type() == PreparedMigDeviceType {
				return true
			}
		}
	}
	return false
}

func (cdi *CDIHandler) DeleteClaimSpecFile(claimUID string) error {
	specName := cdiapi.GenerateTransientSpecName(cdiVendor, cdiClaimClass, claimUID)
	klog.V(6).Infof("Delete CDI spec file: '%s', claim '%s'", specName, claimUID)
	err := os.Remove(filepath.Join(cdi.cdiRoot, specName+".yaml"))
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Philosophy: all devices to be injected into a container are defined in a
// single, transient CDI spec. This function returns the fully qualified
// identifier for a device defined in that spec. Example:
// k8s.gpu.nvidia.com/claim=dab5ab50-d59a-42a6-af16-cfd4628c0f7a-gpu-0
// That identifier can be used elsewhere, and _points to the spec_.
func (cdi *CDIHandler) GetClaimDeviceName(claimUID string, device *AllocatableDevice, containerEdits *cdiapi.ContainerEdits) string {
	return cdiparser.QualifiedName(cdiVendor, cdiClaimClass, fmt.Sprintf("%s-%s", claimUID, device.CanonicalName()))
}

// Construct and return the CDI `deviceNodes` specification for the two
// character devices `/dev/nvidia-caps/nvidia-cap<CIm>` and
// `/dev/nvidia-caps/nvidia-cap<GIm>` for a specific MIG device.
//
// Context: for containerized workload to see and use a specific MIG device, it
// needs to be able to open three character device nodes:
//
// 1) `/dev/nvidia<Pm>`, with <Pm> referring to the parent's minor. This exists
// on the host.
//
// 2) /dev/nvidia-caps/nvidia-cap<CIm> and /dev/nvidia-caps/nvidia-cap<GIm>,
// with <GIm> and <CIm> referring to the MIG GPU instance's and Compute
// instance's minor, respectively. For the the latter two device nodes it is
// sufficient to create them in the container (with proper cgroups permissions),
// without actually requiring the same device nodes to be explicitly created on
// the host. That is what is achieved below with the structure created in
// cdiDevNodeFromNVCapDevInfo().
func (cdi *CDIHandler) GetDevNodesForMigDevice(mlt *MigLiveTuple) ([]*cdispec.DeviceNode, error) {
	gipath, cipath := common.MigCapsPaths(mlt.ParentMinor, mlt.GIID, mlt.CIID)

	giCapsInfo, err := common.ParseNVCapDeviceInfo(gipath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse GI capabilities file %s: %w", gipath, err)
	}

	ciCapsInfo, err := common.ParseNVCapDeviceInfo(cipath)
	if err != nil {
		return nil, fmt.Errorf("failed to parse CI capabilities file %s: %w", cipath, err)
	}

	devnodes := []*cdispec.DeviceNode{giCapsInfo.CDICharDevNode(), ciCapsInfo.CDICharDevNode()}
	return devnodes, nil
}

// Write CDI spec to the filesystem.
func (cdi *CDIHandler) writeSpec(spec spec.Interface, specName string) error {
	// Transform the spec to make it aware that it is running inside a container.
	err := transformroot.New(
		transformroot.WithRoot(cdi.driverRoot),
		transformroot.WithTargetRoot(cdi.targetDriverRoot),
		transformroot.WithRelativeTo("host"),
	).Transform(spec.Raw())
	if err != nil {
		return fmt.Errorf("failed to transform driver root in CDI spec: %w", err)
	}

	klog.V(7).Infof("Write CDI spec: %s", specName)
	return spec.Save(filepath.Join(cdi.cdiRoot, specName+".yaml"))
}
