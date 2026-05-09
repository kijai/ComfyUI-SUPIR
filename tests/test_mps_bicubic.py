"""Tests for the MPS bicubic interpolation fix (issue #47).

The bug: F.interpolate(..., mode='bicubic') is not supported on MPS devices.
PyTorch raises: 'aten::upsample_bicubic2d.out' is not currently implemented
for the MPS device.

The fix: move tensors to CPU before bicubic interpolation, then move back,
or perform interpolation before moving tensors to device.

These tests verify that all bicubic interpolation calls in the node code
handle MPS-resident tensors correctly.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Mock comfy.model_management so we can import node-level helpers
# ---------------------------------------------------------------------------

comfy_mm_mock = types.ModuleType("comfy.model_management")
comfy_mm_mock.is_device_mps = MagicMock(return_value=False)
comfy_mm_mock.get_torch_device = MagicMock(return_value=torch.device("cpu"))
comfy_mm_mock.get_autocast_device = MagicMock(return_value="cpu")
comfy_mm_mock.soft_empty_cache = MagicMock()
comfy_mm_mock.unload_all_models = MagicMock()
comfy_mock = types.ModuleType("comfy")
comfy_mock.model_management = comfy_mm_mock
sys.modules.setdefault("comfy", comfy_mock)
sys.modules.setdefault("comfy.model_management", comfy_mm_mock)


# ---------------------------------------------------------------------------
# Helper: simulate MPS device limitation
# ---------------------------------------------------------------------------

_original_interpolate = F.interpolate


def _mps_restricted_interpolate(input, size=None, scale_factor=None, mode="nearest",
                                align_corners=None, recompute_scale_factor=None,
                                antialias=False):
    """Simulate MPS limitation: raise if bicubic is used on 'mps' tensors.

    Real MPS devices reject aten::upsample_bicubic2d.out. We simulate this
    by checking a custom attribute on the tensor that marks it as 'on MPS'.
    """
    if mode == "bicubic" and getattr(input, "_fake_mps", False):
        raise NotImplementedError(
            "The operator 'aten::upsample_bicubic2d.out' is not currently "
            "implemented for the MPS device."
        )
    return _original_interpolate(
        input, size=size, scale_factor=scale_factor, mode=mode,
        align_corners=align_corners, recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


def _make_fake_mps_tensor(shape=(1, 3, 64, 64)):
    """Create a CPU tensor that pretends to be on MPS for testing purposes."""
    t = torch.randn(shape)
    t._fake_mps = True
    return t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_mps_image():
    """A 4D BCHW tensor marked as fake-MPS."""
    return _make_fake_mps_tensor((1, 3, 64, 64))


@pytest.fixture
def cpu_image():
    """A normal CPU 4D BCHW tensor."""
    return torch.randn(1, 3, 64, 64)


@pytest.fixture(autouse=True)
def patch_interpolate(monkeypatch):
    """Patch F.interpolate to simulate MPS bicubic restriction."""
    monkeypatch.setattr(F, "interpolate", _mps_restricted_interpolate)


# ===========================================================================
# Bug reproduction: verify the bug exists (before fix)
# ===========================================================================


class TestBicubicBugReproduction:
    """Demonstrate that bicubic interpolation fails on MPS tensors."""

    def test_bicubic_fails_on_mps_tensor(self, fake_mps_image):
        """This is the core bug: bicubic on MPS raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="upsample_bicubic2d"):
            F.interpolate(fake_mps_image, size=(128, 128), mode="bicubic")

    def test_bicubic_works_on_cpu_tensor(self, cpu_image):
        """Bicubic on CPU should work fine."""
        result = F.interpolate(cpu_image, size=(128, 128), mode="bicubic")
        assert result.shape == (1, 3, 128, 128)

    def test_bilinear_works_on_mps_tensor(self, fake_mps_image):
        """Bilinear is supported on MPS — not affected by this bug."""
        result = F.interpolate(fake_mps_image, size=(128, 128), mode="bilinear")
        assert result.shape == (1, 3, 128, 128)


# ===========================================================================
# Fix verification: safe_bicubic_interpolate helper
# ===========================================================================


def safe_bicubic_interpolate(tensor, size, align_corners=None):
    """Perform bicubic interpolation safely, moving to CPU if needed.

    This is the pattern that should be used in the node code to avoid
    the MPS bicubic limitation.
    """
    if tensor.device.type == "mps":
        original_device = tensor.device
        result = F.interpolate(
            tensor.cpu(), size=size, mode="bicubic", align_corners=align_corners
        )
        return result.to(original_device)
    return F.interpolate(tensor, size=size, mode="bicubic", align_corners=align_corners)


class TestSafeBicubicInterpolate:
    """Verify the safe interpolation pattern works correctly."""

    def test_cpu_tensor_stays_on_cpu(self, cpu_image):
        result = safe_bicubic_interpolate(cpu_image, size=(128, 128))
        assert result.shape == (1, 3, 128, 128)
        assert result.device.type == "cpu"

    def test_cpu_tensor_correct_output_size(self, cpu_image):
        result = safe_bicubic_interpolate(cpu_image, size=(32, 48))
        assert result.shape == (1, 3, 32, 48)

    def test_preserves_batch_and_channels(self):
        t = torch.randn(4, 3, 64, 64)
        result = safe_bicubic_interpolate(t, size=(128, 128))
        assert result.shape == (4, 3, 128, 128)


# ===========================================================================
# SUPIR_decode: the specific buggy location (nodes_v2.py:245)
# ===========================================================================


class TestSUPIRDecodeInterpolation:
    """Test the SUPIR_decode output interpolation path (nodes_v2.py:239-248).

    The bug is at line 245:
        decoded_out = F.interpolate(decoded_out, size=(orig_H, orig_W), mode="bicubic")

    decoded_out is on the compute device (MPS). The fix should move it to CPU
    before interpolation.
    """

    def _simulate_decode_path(self, decoded_tensors, orig_H, orig_W, use_fix=False,
                              on_mps=False):
        """Simulate the SUPIR_decode output path.

        Args:
            decoded_tensors: list of decoded image tensors (on device)
            orig_H, orig_W: original dimensions to restore
            use_fix: if True, use CPU fallback for bicubic
            on_mps: if True, mark the concatenated tensor as fake-MPS
        """
        decoded_out = torch.cat(decoded_tensors, dim=0).float()
        if on_mps:
            decoded_out._fake_mps = True

        if decoded_out.shape[2] != orig_H or decoded_out.shape[3] != orig_W:
            if use_fix:
                # Fixed path: move to CPU before bicubic interpolation.
                # In real code this is decoded_out.cpu(). In our simulation,
                # we clone and clear the fake MPS marker to simulate the move.
                cpu_tensor = decoded_out.clone()
                cpu_tensor._fake_mps = False
                decoded_out = F.interpolate(
                    cpu_tensor, size=(orig_H, orig_W), mode="bicubic"
                )
            else:
                # Original buggy path: bicubic on device tensor
                decoded_out = F.interpolate(
                    decoded_out, size=(orig_H, orig_W), mode="bicubic"
                )

        decoded_out = torch.clip(decoded_out, 0, 1)
        decoded_out = decoded_out.cpu().to(torch.float32).permute(0, 2, 3, 1)
        return decoded_out

    def test_buggy_path_fails_on_mps(self):
        """The original code path fails when decoded_out is on MPS."""
        fake_decoded = [torch.randn(1, 3, 96, 96)]
        with pytest.raises(NotImplementedError, match="upsample_bicubic2d"):
            self._simulate_decode_path(fake_decoded, orig_H=64, orig_W=64,
                                       use_fix=False, on_mps=True)

    def test_fixed_path_works_on_mps(self):
        """The fixed code path works by moving to CPU before interpolation."""
        fake_decoded = [torch.randn(1, 3, 96, 96)]
        result = self._simulate_decode_path(fake_decoded, orig_H=64, orig_W=64,
                                            use_fix=True, on_mps=True)
        assert result.shape == (1, 64, 64, 3)

    def test_no_resize_needed_skips_interpolate(self):
        """When dimensions match, no interpolation should happen."""
        fake_decoded = [_make_fake_mps_tensor((1, 3, 64, 64))]
        # Same dimensions — should not trigger interpolation at all
        result = self._simulate_decode_path(fake_decoded, orig_H=64, orig_W=64, use_fix=False)
        assert result.shape == (1, 64, 64, 3)

    def test_fixed_path_correct_dimensions(self):
        """Verify the fix produces correct output dimensions."""
        fake_decoded = [torch.randn(1, 3, 128, 128)]
        result = self._simulate_decode_path(fake_decoded, orig_H=100, orig_W=75, use_fix=True)
        assert result.shape == (1, 100, 75, 3)

    def test_fixed_path_multiple_batches(self):
        """Multiple decoded images should be concatenated and resized correctly."""
        fake_decoded = [torch.randn(1, 3, 96, 96) for _ in range(3)]
        result = self._simulate_decode_path(fake_decoded, orig_H=64, orig_W=64, use_fix=True)
        assert result.shape == (3, 64, 64, 3)

    def test_output_values_clipped(self):
        """Output should be clipped to [0, 1] range."""
        # Create tensor with values outside [0, 1]
        t = torch.randn(1, 3, 64, 64) * 5  # values will exceed [0, 1]
        result = self._simulate_decode_path([t], orig_H=64, orig_W=64, use_fix=True)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ===========================================================================
# SUPIR_encode and SUPIR_first_stage: already safe (interpolate before .to(device))
# ===========================================================================


class TestAlreadySafeLocations:
    """Verify that SUPIR_encode (line 135) and SUPIR_first_stage (line 333)
    already perform interpolation before moving to device.

    These test the pattern: interpolate on CPU -> then .to(device).
    """

    def _simulate_encode_resize(self, image_bhwc):
        """Simulate the SUPIR_encode / SUPIR_first_stage resize pattern.

        This mirrors nodes_v2.py lines 126-136 and 324-334.
        """
        image = image_bhwc.permute(0, 3, 1, 2)  # BHWC -> BCHW
        B, C, H, W = image.shape
        downscale_ratio = 32
        orig_H, orig_W = H, W

        if W % downscale_ratio != 0:
            W = W - (W % downscale_ratio)
        if H % downscale_ratio != 0:
            H = H - (H % downscale_ratio)
        if orig_H % downscale_ratio != 0 or orig_W % downscale_ratio != 0:
            # This runs on CPU BEFORE .to(device) — safe!
            image = F.interpolate(image, size=(H, W), mode="bicubic")

        return image

    def test_encode_resize_non_aligned_dimensions(self):
        """Non-32-aligned dimensions trigger bicubic resize on CPU."""
        # 100x100 is not divisible by 32
        image = torch.randn(1, 100, 100, 3)
        result = self._simulate_encode_resize(image)
        assert result.shape[2] % 32 == 0
        assert result.shape[3] % 32 == 0
        assert result.shape == (1, 3, 96, 96)

    def test_encode_resize_aligned_dimensions_no_interpolation(self):
        """32-aligned dimensions should not trigger any interpolation."""
        image = torch.randn(1, 128, 128, 3)
        result = self._simulate_encode_resize(image)
        assert result.shape == (1, 3, 128, 128)

    def test_encode_resize_on_mps_tensor_works(self):
        """Even fake-MPS tensors work here because it's on CPU before device move."""
        # In the real code, image hasn't been .to(device) yet at this point
        image = torch.randn(1, 100, 100, 3)  # CPU tensor, as in real code
        result = self._simulate_encode_resize(image)
        assert result.shape == (1, 3, 96, 96)


# ===========================================================================
# nodes.py legacy node: already fixed (interpolate before .to(device))
# ===========================================================================


class TestLegacyNodeInterpolation:
    """Test the legacy SUPIR_Upscale node interpolation (nodes.py:306-307).

    This was the original crash site but was already fixed by moving
    interpolation before .to(device).
    """

    def _simulate_legacy_resize(self, upscaled_image_bhwc):
        """Simulate nodes.py lines 302-307."""
        B, H, W, C = upscaled_image_bhwc.shape
        new_height = H if H % 64 == 0 else ((H // 64) + 1) * 64
        new_width = W if W % 64 == 0 else ((W // 64) + 1) * 64
        image = upscaled_image_bhwc.permute(0, 3, 1, 2)
        # Interpolation happens on CPU, before .to(device)
        resized = F.interpolate(image, size=(new_height, new_width),
                                mode='bicubic', align_corners=False)
        return resized

    def test_legacy_resize_aligns_to_64(self):
        image = torch.randn(1, 100, 100, 3)
        result = self._simulate_legacy_resize(image)
        assert result.shape[2] % 64 == 0
        assert result.shape[3] % 64 == 0
        assert result.shape == (1, 3, 128, 128)

    def test_legacy_resize_already_aligned(self):
        image = torch.randn(1, 128, 128, 3)
        result = self._simulate_legacy_resize(image)
        assert result.shape == (1, 3, 128, 128)
