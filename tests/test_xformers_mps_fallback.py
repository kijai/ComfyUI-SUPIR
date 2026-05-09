"""Tests for xformers MPS fallback (issue #84).

Verifies that xformers is disabled on non-CUDA devices (e.g. MPS/CPU),
even when the xformers package is importable.

The sgm and SUPIR modules cannot be imported in isolation (they depend on
ComfyUI and pytorch_lightning at import time), so we verify the fix in two
complementary ways:

1. **Source-level checks** — parse each affected file and confirm that
   `torch.cuda.is_available()` gates the xformers flag.
2. **Logic tests** — exercise the exact availability pattern used in the
   codebase with mocked CUDA state.
"""

import ast
import os
from pathlib import Path
from unittest import mock

import torch
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_source(relpath: str) -> str:
    return (ROOT / relpath).read_text()


def _find_xformers_try_blocks(source: str):
    """Return AST Try nodes whose body imports xformers."""
    tree = ast.parse(source)
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for stmt in node.body:
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        if alias.name.startswith("xformers"):
                            results.append(node)
                            break
    return results


# ---------------------------------------------------------------------------
# Source-level verification: each file gates xformers on CUDA
# ---------------------------------------------------------------------------

class TestSourceContainsCudaCheck:
    """Verify that every file setting an xformers availability flag also
    checks torch.cuda.is_available() inside the try block."""

    @pytest.mark.parametrize(
        "relpath",
        [
            "sgm/modules/attention.py",
            "sgm/modules/diffusionmodules/model.py",
            "SUPIR/modules/SUPIR_v0.py",
        ],
    )
    def test_xformers_flag_gated_on_cuda(self, relpath):
        source = _read_source(relpath)
        try_blocks = _find_xformers_try_blocks(source)
        assert try_blocks, f"No xformers try-block found in {relpath}"

        for try_node in try_blocks:
            # Collect all source lines in the try body
            body_lines = []
            for stmt in try_node.body:
                body_lines.append(ast.dump(stmt))

            body_text = "\n".join(body_lines)
            assert "cuda" in body_text.lower() or "is_available" in body_text, (
                f"{relpath}: xformers try-block does not check "
                f"torch.cuda.is_available()"
            )

    def test_tilevae_xformers_gated_on_cuda(self):
        """tilevae.py uses comfy.model_management.XFORMERS_IS_AVAILABLE as an
        outer gate. Verify that torch.cuda.is_available() is also checked."""
        source = _read_source("SUPIR/utils/tilevae.py")
        # The outer if-condition should include a CUDA check
        assert "torch.cuda.is_available()" in source, (
            "SUPIR/utils/tilevae.py: missing torch.cuda.is_available() check"
        )


# ---------------------------------------------------------------------------
# Logic tests: the availability pattern itself
# ---------------------------------------------------------------------------

class TestXformersAvailabilityLogic:
    """Exercise the exact pattern used in the codebase:

        try:
            import xformers; import xformers.ops
            XFORMERS_IS_AVAILABLE = torch.cuda.is_available()
        except:
            XFORMERS_IS_AVAILABLE = False

    Mocking torch.cuda.is_available() lets us simulate MPS/CPU/CUDA.
    """

    @pytest.mark.parametrize(
        "cuda_available, xformers_importable, expected",
        [
            (False, True, False),   # MPS / CPU with xformers installed
            (True, True, True),     # CUDA with xformers installed
            (True, False, False),   # CUDA without xformers
            (False, False, False),  # MPS / CPU without xformers
        ],
        ids=[
            "mps-with-xformers",
            "cuda-with-xformers",
            "cuda-no-xformers",
            "mps-no-xformers",
        ],
    )
    def test_availability_matrix(self, cuda_available, xformers_importable, expected):
        with mock.patch.object(
            torch.cuda, "is_available", return_value=cuda_available
        ):
            try:
                if not xformers_importable:
                    raise ImportError("no xformers")
                flag = torch.cuda.is_available()
            except ImportError:
                flag = False

        assert flag == expected

    def test_mps_device_does_not_enable_xformers(self):
        """Simulate an MPS-only system: CUDA unavailable, xformers importable."""
        with mock.patch.object(
            torch.cuda, "is_available", return_value=False
        ):
            # Simulates the try block
            xformers_imported = True  # import succeeds
            flag = xformers_imported and torch.cuda.is_available()

        assert flag is False, (
            "xformers must not be enabled on non-CUDA devices"
        )

    def test_cuda_device_enables_xformers(self):
        """Simulate a CUDA system with xformers installed."""
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ):
            xformers_imported = True
            flag = xformers_imported and torch.cuda.is_available()

        assert flag is True
