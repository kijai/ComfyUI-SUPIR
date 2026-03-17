"""Tests for the MPS attention buffer size fix (issue #151).

The bug: F.scaled_dot_product_attention() fails with
"RuntimeError: Invalid buffer size: 19.38 GiB" on MPS devices when processing
large images (e.g. 1536x2688). The attention matrix for 64512 spatial tokens
exceeds MPS buffer limits.

The fix: _split_sdp_attention() chunks queries into smaller batches, and
CrossAttention.forward detects MPS + large sequences to use it automatically.
"""

import ast
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
ATTENTION_PATH = ROOT / "sgm" / "modules" / "attention.py"


# ---------------------------------------------------------------------------
# Load _split_sdp_attention and _MPS_ATTENTION_SPLIT_THRESHOLD from source
# without triggering the full module import chain (which needs ComfyUI).
# ---------------------------------------------------------------------------

def _load_split_attention():
    """Extract _split_sdp_attention and threshold from attention.py source."""
    source = ATTENTION_PATH.read_text()

    ns = {"torch": torch, "F": F}
    # Extract the constant and function
    func_lines = []
    capturing = False
    for line in source.splitlines():
        if line.startswith("_MPS_ATTENTION_SPLIT_THRESHOLD"):
            func_lines.append(line)
        elif line.startswith("def _split_sdp_attention"):
            capturing = True
        if capturing:
            func_lines.append(line)
            # Stop at next top-level def/class
            if len(func_lines) > 1 and (line.startswith("def ") or line.startswith("class ")):
                if not line.startswith("def _split_sdp_attention"):
                    func_lines.pop()  # remove the next def line
                    capturing = False

    exec("\n".join(func_lines), ns)
    return ns["_split_sdp_attention"], ns["_MPS_ATTENTION_SPLIT_THRESHOLD"]


_split_sdp_attention, _MPS_ATTENTION_SPLIT_THRESHOLD = _load_split_attention()


# ---------------------------------------------------------------------------
# Source-level checks
# ---------------------------------------------------------------------------


class TestSourceContainsFix:
    """Verify that attention.py contains the MPS buffer size fix."""

    def test_split_attention_function_exists(self):
        source = ATTENTION_PATH.read_text()
        assert "def _split_sdp_attention" in source

    def test_threshold_constant_exists(self):
        source = ATTENTION_PATH.read_text()
        assert "_MPS_ATTENTION_SPLIT_THRESHOLD" in source

    def test_mps_check_in_cross_attention_forward(self):
        """CrossAttention.forward must check for MPS device."""
        source = ATTENTION_PATH.read_text()
        # Find the CrossAttention class and check it references MPS
        in_cross_attention = False
        found_mps_check = False
        for line in source.splitlines():
            if "class CrossAttention" in line:
                in_cross_attention = True
            elif in_cross_attention and line.startswith("class "):
                break
            if in_cross_attention and 'mps' in line.lower():
                found_mps_check = True
        assert found_mps_check, (
            "CrossAttention must check for MPS device to use split attention"
        )

    def test_split_attention_called_for_mps(self):
        """The CrossAttention.forward method must call _split_sdp_attention."""
        source = ATTENTION_PATH.read_text()
        in_cross_attention = False
        found_split_call = False
        for line in source.splitlines():
            if "class CrossAttention" in line:
                in_cross_attention = True
            elif in_cross_attention and line.startswith("class "):
                break
            if in_cross_attention and "_split_sdp_attention" in line:
                found_split_call = True
        assert found_split_call, (
            "CrossAttention.forward must call _split_sdp_attention for MPS"
        )


# ---------------------------------------------------------------------------
# Unit tests for _split_sdp_attention
# ---------------------------------------------------------------------------


class TestSplitSdpAttention:
    """Tests for the _split_sdp_attention helper function."""

    def test_small_sequence_uses_standard_sdp(self):
        """Sequences smaller than chunk_size should use standard SDP."""
        q = torch.randn(1, 2, 64, 32)
        k = torch.randn(1, 2, 64, 32)
        v = torch.randn(1, 2, 64, 32)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = _split_sdp_attention(q, k, v, chunk_size=2048)

        torch.testing.assert_close(result, expected)

    def test_large_sequence_produces_correct_output(self):
        """Chunked attention must produce same result as full attention."""
        seq_len = 256
        q = torch.randn(1, 2, seq_len, 32)
        k = torch.randn(1, 2, seq_len, 32)
        v = torch.randn(1, 2, seq_len, 32)

        expected = F.scaled_dot_product_attention(q, k, v)
        # Use small chunk_size to force chunking
        result = _split_sdp_attention(q, k, v, chunk_size=64)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_chunking_with_non_divisible_sequence(self):
        """Handles sequence lengths not evenly divisible by chunk_size."""
        seq_len = 100  # not divisible by 32
        q = torch.randn(1, 2, seq_len, 32)
        k = torch.randn(1, 2, seq_len, 32)
        v = torch.randn(1, 2, seq_len, 32)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = _split_sdp_attention(q, k, v, chunk_size=32)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_output_shape_matches_query(self):
        """Output shape must match query shape."""
        q = torch.randn(2, 4, 128, 64)
        k = torch.randn(2, 4, 128, 64)
        v = torch.randn(2, 4, 128, 64)

        result = _split_sdp_attention(q, k, v, chunk_size=32)
        assert result.shape == q.shape

    def test_cross_attention_different_kv_length(self):
        """Cross-attention: query and key/value can have different lengths."""
        q = torch.randn(1, 2, 128, 32)
        k = torch.randn(1, 2, 77, 32)  # text tokens
        v = torch.randn(1, 2, 77, 32)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = _split_sdp_attention(q, k, v, chunk_size=32)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_with_attention_mask(self):
        """Attention mask must be chunked along the query dimension."""
        seq_len = 64
        q = torch.randn(1, 2, seq_len, 32)
        k = torch.randn(1, 2, seq_len, 32)
        v = torch.randn(1, 2, seq_len, 32)
        mask = torch.zeros(1, 2, seq_len, seq_len)

        expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        result = _split_sdp_attention(q, k, v, attn_mask=mask, chunk_size=16)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_single_element_chunks(self):
        """Extreme case: chunk_size=1 should still produce correct results."""
        q = torch.randn(1, 1, 8, 16)
        k = torch.randn(1, 1, 8, 16)
        v = torch.randn(1, 1, 8, 16)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = _split_sdp_attention(q, k, v, chunk_size=1)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_batch_and_heads_preserved(self):
        """Multiple batches and heads must be handled correctly."""
        B, H, N, D = 3, 8, 64, 32
        q = torch.randn(B, H, N, D)
        k = torch.randn(B, H, N, D)
        v = torch.randn(B, H, N, D)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = _split_sdp_attention(q, k, v, chunk_size=16)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Threshold logic tests
# ---------------------------------------------------------------------------


class TestThresholdLogic:
    """Tests for the MPS split-attention threshold logic."""

    def test_threshold_is_reasonable(self):
        """Threshold should be between 64M and 1B elements per head."""
        assert 64 * 1024 * 1024 <= _MPS_ATTENTION_SPLIT_THRESHOLD <= 1024 * 1024 * 1024

    def test_large_image_exceeds_threshold(self):
        """A 1536x2688 image (the bug report dimensions) must exceed threshold."""
        # Latent is 1/8 of pixel size: 192x336 = 64512 tokens
        latent_h, latent_w = 192, 336
        seq_len = latent_h * latent_w  # 64512
        n_elements = seq_len * seq_len  # self-attention: q_len == k_len
        assert n_elements > _MPS_ATTENTION_SPLIT_THRESHOLD, (
            f"1536x2688 image ({seq_len} tokens, {n_elements} elements) "
            f"must exceed threshold {_MPS_ATTENTION_SPLIT_THRESHOLD}"
        )

    def test_small_image_below_threshold(self):
        """A 512x512 image should be below the threshold (no chunking needed)."""
        latent_h, latent_w = 64, 64
        seq_len = latent_h * latent_w  # 4096
        n_elements = seq_len * seq_len  # 16M
        assert n_elements < _MPS_ATTENTION_SPLIT_THRESHOLD, (
            f"512x512 image ({seq_len} tokens) should not trigger chunking"
        )

    @pytest.mark.parametrize(
        "pixel_h, pixel_w, should_exceed",
        [
            (512, 512, False),     # 4096 tokens, 16M elements
            (1024, 1024, False),   # 16384 tokens, ~268M elements
            (1536, 2688, True),    # 64512 tokens, ~4.2B elements (the bug report)
            (2048, 2048, True),    # 65536 tokens, ~4.3B elements
            (768, 768, False),     # 9216 tokens, ~85M elements
        ],
        ids=[
            "512x512-no-split",
            "1024x1024-no-split",
            "1536x2688-must-split",
            "2048x2048-must-split",
            "768x768-no-split",
        ],
    )
    def test_various_image_sizes(self, pixel_h, pixel_w, should_exceed):
        latent_h, latent_w = pixel_h // 8, pixel_w // 8
        seq_len = latent_h * latent_w
        n_elements = seq_len * seq_len
        if should_exceed:
            assert n_elements > _MPS_ATTENTION_SPLIT_THRESHOLD
        else:
            assert n_elements <= _MPS_ATTENTION_SPLIT_THRESHOLD


# ---------------------------------------------------------------------------
# Integration: simulate CrossAttention.forward behavior
# ---------------------------------------------------------------------------


class TestCrossAttentionMPSIntegration:
    """Simulate the CrossAttention.forward logic to verify the MPS path."""

    def _simulate_cross_attention_forward(self, q, k, v, mask=None):
        """Mimic the fixed CrossAttention.forward dispatch logic."""
        n_elements_per_head = q.shape[2] * k.shape[2]
        if q.device.type == "mps" and n_elements_per_head > _MPS_ATTENTION_SPLIT_THRESHOLD:
            return _split_sdp_attention(q, k, v, attn_mask=mask)
        else:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    def test_cpu_uses_standard_attention(self):
        """CPU tensors should always use standard SDP."""
        q = torch.randn(1, 2, 256, 32)
        k = torch.randn(1, 2, 256, 32)
        v = torch.randn(1, 2, 256, 32)

        # Even with huge "sequence", CPU doesn't trigger split
        expected = F.scaled_dot_product_attention(q, k, v)
        result = self._simulate_cross_attention_forward(q, k, v)
        torch.testing.assert_close(result, expected)

    def test_split_produces_same_result_as_full(self):
        """Split attention must be numerically equivalent to full attention."""
        seq_len = 128
        q = torch.randn(1, 2, seq_len, 32)
        k = torch.randn(1, 2, seq_len, 32)
        v = torch.randn(1, 2, seq_len, 32)

        full = F.scaled_dot_product_attention(q, k, v)
        split = _split_sdp_attention(q, k, v, chunk_size=32)

        torch.testing.assert_close(split, full, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Bug reproduction scenario
# ---------------------------------------------------------------------------


class TestBugReproduction:
    """Reproduce the exact scenario from issue #151.

    Error: RuntimeError: Invalid buffer size: 19.38 GiB
    Image: 1536x2688 -> latent 192x336 -> 64512 spatial tokens
    Attention matrix: 64512 x 64512 x 4 bytes ≈ 15.5 GiB per head
    """

    def test_large_image_attention_with_chunking(self):
        """The problematic sequence length must work with split attention."""
        # Use a scaled-down version to keep tests fast, but verify the
        # chunking logic handles the proportions correctly
        # Real: 64512 tokens. Test: 1024 tokens with small chunk_size
        seq_len = 1024
        q = torch.randn(1, 2, seq_len, 32)
        k = torch.randn(1, 2, seq_len, 32)
        v = torch.randn(1, 2, seq_len, 32)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = _split_sdp_attention(q, k, v, chunk_size=128)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_issue_151_dimensions_would_trigger_split(self):
        """The exact dimensions from issue #151 must trigger split on MPS."""
        latent_h, latent_w = 192, 336  # 1536x2688 / 8
        seq_len = latent_h * latent_w   # 64512
        n_elements = seq_len * seq_len  # ~4.16 billion

        assert n_elements > _MPS_ATTENTION_SPLIT_THRESHOLD, (
            f"Issue #151 dimensions ({seq_len} tokens) must trigger "
            f"split attention on MPS"
        )

    @pytest.mark.parametrize("chunk_size", [128, 256, 512, 1024, 2048])
    def test_various_chunk_sizes_produce_same_result(self, chunk_size):
        """All chunk sizes must produce the same result."""
        seq_len = 256
        q = torch.randn(1, 2, seq_len, 32)
        k = torch.randn(1, 2, seq_len, 32)
        v = torch.randn(1, 2, seq_len, 32)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = _split_sdp_attention(q, k, v, chunk_size=chunk_size)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_cfg_doubled_batch(self):
        """CFG doubles the batch size — verify chunking handles this."""
        # Classifier-free guidance: batch=2 (conditional + unconditional)
        seq_len = 256
        q = torch.randn(2, 8, seq_len, 64)  # batch=2, heads=8
        k = torch.randn(2, 8, seq_len, 64)
        v = torch.randn(2, 8, seq_len, 64)

        expected = F.scaled_dot_product_attention(q, k, v)
        result = _split_sdp_attention(q, k, v, chunk_size=64)

        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)
