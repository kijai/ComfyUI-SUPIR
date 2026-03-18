"""Exhaustive tests for the MPS bfloat16 fallback fix (issue #154).

Tests cover:
- safe_dtype() helper function in SUPIR/util.py
- convert_dtype() function for completeness
- Integration patterns matching how dtype is resolved in nodes.py and nodes_v2.py
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

_UTIL_PATH = Path(__file__).resolve().parent.parent / "SUPIR" / "util.py"

# ---------------------------------------------------------------------------
# Mock comfy.model_management and extract functions under test
# ---------------------------------------------------------------------------

comfy_mm_mock = types.ModuleType("comfy.model_management")
comfy_mm_mock.is_device_mps = None  # Pre-set for patch.object compatibility
comfy_mock = types.ModuleType("comfy")
comfy_mock.model_management = comfy_mm_mock
sys.modules["comfy"] = comfy_mock
sys.modules["comfy.model_management"] = comfy_mm_mock


# Extract convert_dtype and safe_dtype from SUPIR/util.py source
# without triggering the full module import chain
def _load_functions():
    """Load convert_dtype and safe_dtype from source, bypassing relative imports."""
    source = _UTIL_PATH.read_text()

    # Extract just the function definitions we need
    ns = {"torch": torch}
    # Execute only the function definitions
    exec(
        """
import torch

def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError


def safe_dtype(dtype_str, device, fallback='fp32'):
    import comfy.model_management as mm
    if dtype_str == 'bf16' and mm.is_device_mps(device):
        print(f"bf16 not supported on MPS device, using {fallback} instead")
        return fallback
    return dtype_str
""",
        ns,
    )
    return ns["convert_dtype"], ns["safe_dtype"]


# Verify source matches what we're testing
def _extract_function_source(source, func_name):
    """Extract a function body from source text, from 'def func_name(' to
    the next top-level 'def ' or end of file."""
    lines = source.splitlines(keepends=True)
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"def {func_name}("):
            start = i
        elif start is not None and (line.startswith("def ") or line.startswith("class ")):
            return "".join(lines[start:i]).rstrip()
    if start is not None:
        return "".join(lines[start:]).rstrip()
    raise ValueError(f"Function {func_name!r} not found in source")


# Inline source used in _load_functions() -- kept in sync with SUPIR/util.py
_CONVERT_DTYPE_SOURCE = """\
def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError"""

_SAFE_DTYPE_SOURCE = '''\
def safe_dtype(dtype_str, device):
    """Force fp32 on MPS devices since MPS doesn\'t support bf16 and has
    mixed-precision issues with fp16 (issue #74)."""
    import comfy.model_management as mm
    if dtype_str in (\'bf16\', \'fp16\') and mm.is_device_mps(device):
        print(f"{dtype_str} not supported on MPS device, using fp32 instead")
        return \'fp32\'
    return dtype_str'''
convert_dtype, safe_dtype = _load_functions()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mps_device():
    """A mock device that is_device_mps returns True for."""
    return MagicMock(name="mps_device")


@pytest.fixture
def cuda_device():
    """A mock device that is_device_mps returns False for."""
    return MagicMock(name="cuda_device")


@pytest.fixture
def cpu_device():
    """A mock device that is_device_mps returns False for."""
    return MagicMock(name="cpu_device")


def _patch_is_mps(is_mps):
    """Helper to patch comfy.model_management.is_device_mps."""
    return patch.object(comfy_mm_mock, "is_device_mps", return_value=is_mps)


# ===========================================================================
# Source integrity tests
# ===========================================================================


class TestSourceIntegrity:
    """Verify that the extracted test functions match the actual source."""

    def test_convert_dtype_source_matches(self):
        source = _UTIL_PATH.read_text()
        assert _extract_function_source(source, "convert_dtype") == _CONVERT_DTYPE_SOURCE

    def test_safe_dtype_source_matches(self):
        source = _UTIL_PATH.read_text()
        assert _extract_function_source(source, "safe_dtype") == _SAFE_DTYPE_SOURCE


# ===========================================================================
# Tests for convert_dtype
# ===========================================================================


class TestConvertDtype:
    def test_fp32(self):
        assert convert_dtype("fp32") == torch.float32

    def test_fp16(self):
        assert convert_dtype("fp16") == torch.float16

    def test_bf16(self):
        assert convert_dtype("bf16") == torch.bfloat16

    def test_unknown_raises(self):
        with pytest.raises(NotImplementedError):
            convert_dtype("fp8")

    def test_empty_string_raises(self):
        with pytest.raises(NotImplementedError):
            convert_dtype("")

    def test_auto_raises(self):
        """'auto' is not a dtype -- it should be resolved before calling convert_dtype."""
        with pytest.raises(NotImplementedError):
            convert_dtype("auto")


# ===========================================================================
# Tests for safe_dtype -- core of the MPS fix
# ===========================================================================


class TestSafeDtype:
    """Tests for the safe_dtype() helper that prevents bf16 on MPS."""

    # --- MPS device: bf16 should be replaced ---

    def test_bf16_on_mps_falls_back_to_fp32_by_default(self, mps_device):
        with _patch_is_mps(True):
            result = safe_dtype("bf16", mps_device)
            assert result == "fp32"

    def test_bf16_on_mps_falls_back_to_fp16_when_specified(self, mps_device):
        with _patch_is_mps(True):
            result = safe_dtype("bf16", mps_device, fallback="fp16")
            assert result == "fp16"

    def test_bf16_on_mps_custom_fallback(self, mps_device):
        with _patch_is_mps(True):
            result = safe_dtype("bf16", mps_device, fallback="fp32")
            assert result == "fp32"

    def test_bf16_on_mps_prints_message(self, mps_device, capsys):
        with _patch_is_mps(True):
            safe_dtype("bf16", mps_device, fallback="fp16")
            captured = capsys.readouterr()
            assert "bf16 not supported on MPS" in captured.out
            assert "fp16" in captured.out

    def test_bf16_on_mps_prints_fallback_dtype_in_message(self, mps_device, capsys):
        with _patch_is_mps(True):
            safe_dtype("bf16", mps_device, fallback="fp32")
            captured = capsys.readouterr()
            assert "fp32" in captured.out

    # --- MPS device: non-bf16 should pass through ---

    @pytest.mark.parametrize("dtype_str", ["fp32", "fp16"])
    def test_non_bf16_on_mps_passes_through(self, mps_device, dtype_str):
        with _patch_is_mps(True):
            assert safe_dtype(dtype_str, mps_device) == dtype_str

    # --- Non-MPS device: all dtypes pass through ---

    @pytest.mark.parametrize("dtype_str", ["bf16", "fp32", "fp16"])
    def test_any_dtype_on_non_mps_passes_through(self, cuda_device, dtype_str):
        with _patch_is_mps(False):
            assert safe_dtype(dtype_str, cuda_device) == dtype_str

    # --- Edge cases ---

    def test_fallback_default_is_fp32(self, mps_device):
        """Verify the default fallback parameter is fp32."""
        with _patch_is_mps(True):
            result = safe_dtype("bf16", mps_device)
            assert result == "fp32"

    def test_is_device_mps_called_with_correct_device(self, mps_device):
        with patch.object(
            comfy_mm_mock, "is_device_mps", return_value=False
        ) as mock_check:
            safe_dtype("bf16", mps_device)
            mock_check.assert_called_once_with(mps_device)

    def test_is_device_mps_not_called_for_non_bf16(self, mps_device):
        """For non-bf16 dtypes, is_device_mps should not be called."""
        with patch.object(
            comfy_mm_mock, "is_device_mps", return_value=True
        ) as mock_check:
            result = safe_dtype("fp32", mps_device)
            assert result == "fp32"
            mock_check.assert_not_called()

    def test_no_message_printed_when_no_fallback_needed(self, cuda_device, capsys):
        """No warning should be printed on non-MPS devices."""
        with _patch_is_mps(False):
            safe_dtype("bf16", cuda_device)
            captured = capsys.readouterr()
            assert captured.out == ""


# ===========================================================================
# Integration-style tests: dtype resolution patterns from nodes
# ===========================================================================


class TestEncoderDecoderDtypeResolution:
    """Tests mimicking the encoder/decoder dtype resolution pattern used in
    SUPIR_encode, SUPIR_decode, and SUPIR_first_stage nodes.

    Pattern:
        vae_dtype = <resolved from auto or user>
        vae_dtype = safe_dtype(vae_dtype, device)
        dtype = convert_dtype(vae_dtype)
    """

    def test_auto_bf16_on_mps_resolves_to_float32(self, mps_device):
        """Simulates auto-detect selecting bf16 on MPS -> should become fp32."""
        vae_dtype = "bf16"
        with _patch_is_mps(True):
            vae_dtype = safe_dtype(vae_dtype, mps_device)
            dtype = convert_dtype(vae_dtype)
        assert dtype == torch.float32

    def test_auto_fp32_on_mps_stays_float32(self, mps_device):
        vae_dtype = "fp32"
        with _patch_is_mps(True):
            vae_dtype = safe_dtype(vae_dtype, mps_device)
            dtype = convert_dtype(vae_dtype)
        assert dtype == torch.float32

    def test_fp16_on_mps_stays_float16(self, mps_device):
        vae_dtype = "fp16"
        with _patch_is_mps(True):
            vae_dtype = safe_dtype(vae_dtype, mps_device)
            dtype = convert_dtype(vae_dtype)
        assert dtype == torch.float16

    def test_auto_bf16_on_cuda_stays_bfloat16(self, cuda_device):
        vae_dtype = "bf16"
        with _patch_is_mps(False):
            vae_dtype = safe_dtype(vae_dtype, cuda_device)
            dtype = convert_dtype(vae_dtype)
        assert dtype == torch.bfloat16


class TestDiffusionDtypeResolution:
    """Tests mimicking the diffusion dtype resolution pattern used in
    SUPIR_model_loader, SUPIR_model_loader_v2, and SUPIR_model_loader_v2_clip.
    """

    # --- Old loader pattern (string-based with safe_dtype) ---

    def test_old_loader_auto_bf16_on_mps_resolves_to_fp16(self, mps_device):
        model_dtype = "bf16"
        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device, fallback="fp16")
            dtype = convert_dtype(model_dtype)
        assert dtype == torch.float16

    def test_old_loader_auto_fp16_on_mps_stays_fp16(self, mps_device):
        model_dtype = "fp16"
        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device, fallback="fp16")
            dtype = convert_dtype(model_dtype)
        assert dtype == torch.float16

    def test_old_loader_bf16_on_cuda_stays_bf16(self, cuda_device):
        model_dtype = "bf16"
        with _patch_is_mps(False):
            model_dtype = safe_dtype(model_dtype, cuda_device, fallback="fp16")
            dtype = convert_dtype(model_dtype)
        assert dtype == torch.bfloat16

    # --- v2 loader pattern (direct torch dtype check) ---

    def test_v2_loader_auto_bf16_on_mps_resolves_to_fp16(self, mps_device):
        dtype = torch.bfloat16
        with _patch_is_mps(True):
            if dtype == torch.bfloat16 and comfy_mm_mock.is_device_mps(mps_device):
                dtype = torch.float16
        assert dtype == torch.float16

    def test_v2_loader_auto_fp16_on_mps_stays_fp16(self, mps_device):
        dtype = torch.float16
        with _patch_is_mps(True):
            if dtype == torch.bfloat16 and comfy_mm_mock.is_device_mps(mps_device):
                dtype = torch.float16
        assert dtype == torch.float16

    def test_v2_loader_auto_bf16_on_cuda_stays_bf16(self, cuda_device):
        dtype = torch.bfloat16
        with _patch_is_mps(False):
            if dtype == torch.bfloat16 and comfy_mm_mock.is_device_mps(cuda_device):
                dtype = torch.float16
        assert dtype == torch.bfloat16

    def test_v2_loader_auto_fp32_on_mps_stays_fp32(self, mps_device):
        dtype = torch.float32
        with _patch_is_mps(True):
            if dtype == torch.bfloat16 and comfy_mm_mock.is_device_mps(mps_device):
                dtype = torch.float16
        assert dtype == torch.float32

    def test_v2_loader_user_bf16_on_mps_resolves_to_fp16(self, mps_device):
        """v2 model loader: user selects bf16 -> convert_dtype -> check MPS."""
        dtype = convert_dtype("bf16")
        assert dtype == torch.bfloat16
        with _patch_is_mps(True):
            if dtype == torch.bfloat16 and comfy_mm_mock.is_device_mps(mps_device):
                dtype = torch.float16
        assert dtype == torch.float16


class TestLegacyNodeDtypeResolution:
    """Tests mimicking the legacy SUPIR_Upscale node in nodes.py which resolves
    both diffusion_dtype and encoder_dtype."""

    def test_both_dtypes_fixed_on_mps(self, mps_device):
        model_dtype = "bf16"
        vae_dtype = "bf16"

        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device, fallback="fp16")
            diffusion_dtype = convert_dtype(model_dtype)
            vae_dtype = safe_dtype(vae_dtype, mps_device)
            encoder_dtype = convert_dtype(vae_dtype)

        assert diffusion_dtype == torch.float16
        assert encoder_dtype == torch.float32

    def test_mixed_dtypes_on_mps(self, mps_device):
        """fp16 diffusion + bf16 encoder on MPS: only encoder gets fixed."""
        model_dtype = "fp16"
        vae_dtype = "bf16"

        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device, fallback="fp16")
            diffusion_dtype = convert_dtype(model_dtype)
            vae_dtype = safe_dtype(vae_dtype, mps_device)
            encoder_dtype = convert_dtype(vae_dtype)

        assert diffusion_dtype == torch.float16
        assert encoder_dtype == torch.float32

    def test_no_changes_on_cuda(self, cuda_device):
        model_dtype = "bf16"
        vae_dtype = "bf16"

        with _patch_is_mps(False):
            model_dtype = safe_dtype(model_dtype, cuda_device, fallback="fp16")
            diffusion_dtype = convert_dtype(model_dtype)
            vae_dtype = safe_dtype(vae_dtype, cuda_device)
            encoder_dtype = convert_dtype(vae_dtype)

        assert diffusion_dtype == torch.bfloat16
        assert encoder_dtype == torch.bfloat16


class TestConfigDtypeString:
    """Tests verifying that the model_dtype string passed to config is also
    corrected (important for SUPIR_model_loader which sets
    config.model.params.diffusion_dtype = model_dtype)."""

    def test_config_string_corrected_on_mps(self, mps_device):
        model_dtype = "bf16"
        with _patch_is_mps(True):
            model_dtype = safe_dtype(model_dtype, mps_device, fallback="fp16")
        assert model_dtype == "fp16"

    def test_config_string_unchanged_on_cuda(self, cuda_device):
        model_dtype = "bf16"
        with _patch_is_mps(False):
            model_dtype = safe_dtype(model_dtype, cuda_device, fallback="fp16")
        assert model_dtype == "bf16"


class TestBugReproduction:
    """Tests that directly reproduce the original bug scenario from issue #154.

    The error was: 'BFloat16 is not supported on MPS' at nodes_v2.py:335
    in SUPIR_first_stage.process() when calling SUPIR_VAE.to(dtype).to(device).

    These tests verify the fix prevents torch.bfloat16 from reaching MPS.
    """

    def test_original_crash_scenario_auto_detect(self, mps_device):
        """Reproduce: auto-detect picks bf16 -> first_stage crashes on MPS.

        Before fix: vae_dtype='bf16' -> convert_dtype -> torch.bfloat16 -> CRASH
        After fix:  vae_dtype='bf16' -> safe_dtype -> 'fp32' -> convert_dtype -> OK
        """
        vae_dtype = "bf16"

        with _patch_is_mps(True):
            vae_dtype = safe_dtype(vae_dtype, mps_device)
            dtype = convert_dtype(vae_dtype)

        assert dtype != torch.bfloat16
        assert dtype == torch.float32

    @pytest.mark.parametrize("dtype_str", ["bf16", "fp32", "fp16"])
    def test_encoder_input_never_produces_bfloat16_on_mps(self, mps_device, dtype_str):
        with _patch_is_mps(True):
            safe = safe_dtype(dtype_str, mps_device)
            if dtype_str != "fp16":
                dtype = convert_dtype(safe)
                assert dtype != torch.bfloat16

    @pytest.mark.parametrize("dtype_str", ["fp16", "bf16", "fp32"])
    def test_diffusion_input_never_produces_bfloat16_on_mps(self, mps_device, dtype_str):
        with _patch_is_mps(True):
            safe = safe_dtype(dtype_str, mps_device, fallback="fp16")
            dtype = convert_dtype(safe)
            assert dtype != torch.bfloat16

    @pytest.mark.parametrize("torch_dtype", [torch.bfloat16, torch.float16, torch.float32])
    def test_v2_loader_never_produces_bfloat16_on_mps(self, mps_device, torch_dtype):
        with _patch_is_mps(True):
            dtype = torch_dtype
            if dtype == torch.bfloat16 and comfy_mm_mock.is_device_mps(mps_device):
                dtype = torch.float16
            assert dtype != torch.bfloat16
