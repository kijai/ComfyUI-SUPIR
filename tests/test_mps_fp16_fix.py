"""Tests for the MPS fp16/bf16 fallback fix (issue #74).

On MPS devices, both fp16 and bf16 cause crashes because:
1. bf16 is not supported at all on MPS
2. fp16 causes mixed-precision crashes — autocast is disabled on MPS,
   so fp16 model weights mix with fp32 intermediate tensors (e.g.,
   denoiser scaling coefficients), causing 'not broadcast compatible' errors

The fix adds safe_dtype() to force fp32 on MPS for both fp16 and bf16.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

_UTIL_PATH = Path(__file__).resolve().parent.parent / "SUPIR" / "util.py"

# ---------------------------------------------------------------------------
# Mock comfy.model_management
# ---------------------------------------------------------------------------

comfy_mm_mock = types.ModuleType("comfy.model_management")
comfy_mm_mock.is_device_mps = None
comfy_mock = types.ModuleType("comfy")
comfy_mock.model_management = comfy_mm_mock
sys.modules.setdefault("comfy", comfy_mock)
sys.modules.setdefault("comfy.model_management", comfy_mm_mock)


def _load_functions():
    """Load convert_dtype and safe_dtype from SUPIR/util.py source."""
    source = _UTIL_PATH.read_text()

    ns = {"torch": torch}
    # Extract function definitions — we exec the tail of the file
    # that contains convert_dtype and safe_dtype
    func_source = ""
    in_func = False
    for line in source.splitlines():
        if line.startswith("def convert_dtype") or line.startswith("def safe_dtype"):
            in_func = True
        if in_func:
            func_source += line + "\n"

    exec(func_source, ns)
    return ns.get("convert_dtype"), ns.get("safe_dtype")


convert_dtype, safe_dtype = _load_functions()


def _patch_is_mps(is_mps):
    return patch.object(comfy_mm_mock, "is_device_mps", return_value=is_mps)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mps_device():
    return MagicMock(name="mps_device")


@pytest.fixture
def cuda_device():
    return MagicMock(name="cuda_device")


# ===========================================================================
# Prerequisite: safe_dtype must exist
# ===========================================================================

class TestSafeDtypeExists:
    """safe_dtype() must be defined in SUPIR/util.py."""

    def test_safe_dtype_is_callable(self):
        assert safe_dtype is not None, (
            "safe_dtype() not found in SUPIR/util.py — "
            "the function must be added to fix issue #74"
        )


# ===========================================================================
# Core: fp16 must fall back to fp32 on MPS (issue #74)
# ===========================================================================

class TestFp16FallbackOnMps:
    """The main bug: fp16 on MPS crashes with mixed-precision errors."""

    def test_fp16_on_mps_falls_back_to_fp32(self, mps_device):
        with _patch_is_mps(True):
            result = safe_dtype("fp16", mps_device)
            assert result == "fp32", (
                "fp16 must fall back to fp32 on MPS — "
                "MPS can't handle fp16 mixed precision (issue #74)"
            )

    def test_fp16_on_mps_prints_warning(self, mps_device, capsys):
        with _patch_is_mps(True):
            safe_dtype("fp16", mps_device)
            captured = capsys.readouterr()
            assert "fp16" in captured.out.lower()
            assert "mps" in captured.out.lower()

    def test_fp16_on_mps_converts_to_float32(self, mps_device):
        """End-to-end: fp16 string → safe_dtype → convert_dtype → torch.float32."""
        with _patch_is_mps(True):
            safe = safe_dtype("fp16", mps_device)
            dtype = convert_dtype(safe)
            assert dtype == torch.float32

    def test_fp16_on_cuda_stays_fp16(self, cuda_device):
        with _patch_is_mps(False):
            result = safe_dtype("fp16", cuda_device)
            assert result == "fp16"


# ===========================================================================
# bf16 must also fall back to fp32 on MPS
# ===========================================================================

class TestBf16FallbackOnMps:
    """bf16 is not supported on MPS at all."""

    def test_bf16_on_mps_falls_back_to_fp32(self, mps_device):
        with _patch_is_mps(True):
            result = safe_dtype("bf16", mps_device)
            assert result == "fp32"

    def test_bf16_on_cuda_stays_bf16(self, cuda_device):
        with _patch_is_mps(False):
            result = safe_dtype("bf16", cuda_device)
            assert result == "bf16"


# ===========================================================================
# fp32 should pass through on all devices
# ===========================================================================

class TestFp32Passthrough:
    def test_fp32_on_mps_stays_fp32(self, mps_device):
        with _patch_is_mps(True):
            assert safe_dtype("fp32", mps_device) == "fp32"

    def test_fp32_on_cuda_stays_fp32(self, cuda_device):
        with _patch_is_mps(False):
            assert safe_dtype("fp32", cuda_device) == "fp32"


# ===========================================================================
# Integration: dtype resolution patterns from nodes
# ===========================================================================

class TestDiffusionDtypeResolution:
    """Mimics the diffusion dtype resolution in node loaders."""

    def test_auto_detect_fp16_on_mps_resolves_to_fp32(self, mps_device):
        """Simulates auto-detect selecting fp16 on MPS."""
        model_dtype = "fp16"  # auto-detect picked fp16
        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device)
            dtype = convert_dtype(model_dtype)
        assert dtype == torch.float32

    def test_auto_detect_bf16_on_mps_resolves_to_fp32(self, mps_device):
        model_dtype = "bf16"
        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device)
            dtype = convert_dtype(model_dtype)
        assert dtype == torch.float32

    def test_user_selects_fp16_on_mps_resolves_to_fp32(self, mps_device):
        model_dtype = "fp16"
        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device)
            dtype = convert_dtype(model_dtype)
        assert dtype == torch.float32


class TestEncoderDtypeResolution:
    """Mimics encoder/decoder dtype resolution in SUPIR_encode/decode/first_stage."""

    def test_encoder_bf16_on_mps_resolves_to_fp32(self, mps_device):
        vae_dtype = "bf16"
        with _patch_is_mps(True):
            vae_dtype = safe_dtype(vae_dtype, mps_device)
            dtype = convert_dtype(vae_dtype)
        assert dtype == torch.float32

    def test_encoder_fp16_on_mps_resolves_to_fp32(self, mps_device):
        """fp16 encoder would also crash on MPS."""
        vae_dtype = "fp16"
        with _patch_is_mps(True):
            vae_dtype = safe_dtype(vae_dtype, mps_device)
            dtype = convert_dtype(vae_dtype)
        assert dtype == torch.float32


class TestV2LoaderDtypeResolution:
    """Mimics the v2 model loader pattern that uses direct torch dtypes."""

    @pytest.mark.parametrize("torch_dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_on_mps_resolves_to_fp32(self, mps_device, torch_dtype):
        """Both fp16 and bf16 must become fp32 on MPS."""
        dtype = torch_dtype
        with _patch_is_mps(True):
            if (dtype == torch.float16 or dtype == torch.bfloat16) and comfy_mm_mock.is_device_mps(mps_device):
                dtype = torch.float32
        assert dtype == torch.float32

    def test_fp32_on_mps_stays_fp32(self, mps_device):
        dtype = torch.float32
        with _patch_is_mps(True):
            if (dtype == torch.float16 or dtype == torch.bfloat16) and comfy_mm_mock.is_device_mps(mps_device):
                dtype = torch.float32
        assert dtype == torch.float32

    @pytest.mark.parametrize("torch_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_any_dtype_on_cuda_unchanged(self, cuda_device, torch_dtype):
        dtype = torch_dtype
        with _patch_is_mps(False):
            if (dtype == torch.float16 or dtype == torch.bfloat16) and comfy_mm_mock.is_device_mps(cuda_device):
                dtype = torch.float32
        assert dtype == torch_dtype


# ===========================================================================
# Bug reproduction: the exact crash scenario from issue #74
# ===========================================================================

class TestBugReproduction:
    """Reproduce the exact scenario from issue #74.

    Error: input types 'tensor<2x1280xf32>' and 'tensor<1280xf16>' are not
    broadcast compatible.

    This happens in the denoiser when scaling coefficients (fp32) are
    multiplied with model tensors (fp16) on MPS without autocast.
    """

    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16"])
    def test_no_half_precision_reaches_mps(self, mps_device, dtype_str):
        """After safe_dtype, no half-precision dtype should reach MPS."""
        with _patch_is_mps(True):
            safe = safe_dtype(dtype_str, mps_device)
            dtype = convert_dtype(safe)
            assert dtype == torch.float32, (
                f"{dtype_str} was not converted to fp32 for MPS — "
                f"this will crash with mixed-precision errors"
            )

    def test_legacy_node_both_dtypes_safe_on_mps(self, mps_device):
        """Legacy SUPIR_Upscale node sets both diffusion and encoder dtype."""
        model_dtype = "fp16"
        vae_dtype = "bf16"

        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device)
            diffusion_dtype = convert_dtype(model_dtype)
            vae_dtype = safe_dtype(vae_dtype, mps_device)
            encoder_dtype = convert_dtype(vae_dtype)

        assert diffusion_dtype == torch.float32
        assert encoder_dtype == torch.float32

    def test_config_dtype_string_corrected_on_mps(self, mps_device):
        """The string dtype passed to model config must also be corrected."""
        model_dtype = "fp16"
        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device)
        assert model_dtype == "fp32", (
            "Config string must be 'fp32' on MPS so the model "
            "initializes with correct dtype"
        )
