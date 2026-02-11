"""A/B comparison test: natten (native) vs pure-PyTorch fallback.

Run on a machine where natten is installed (e.g. macOS) to verify numerical
equivalence before relying on the fallback on Windows.

Usage:
    cd backend && uv run python test_natten_compat.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Step 1 – import the REAL natten functions BEFORE any fallback is registered.
from allin1_enhancer import _ensure_natten_torch_compat
_ensure_natten_torch_compat()

import torch
from natten.functional import natten1dqkrpb as _orig_1d_qkrpb  # type: ignore
from natten.functional import natten1dav as _orig_1d_av  # type: ignore
from natten.functional import natten2dqkrpb as _orig_2d_qkrpb  # type: ignore
from natten.functional import natten2dav as _orig_2d_av  # type: ignore

# Step 2 – forcefully install the pure-PyTorch fallback under a different name
# so we can compare both.
# We temporarily remove natten from sys.modules, install the fallback,
# then restore the real natten.
_saved_natten = sys.modules.pop("natten")
_saved_nf = sys.modules.pop("natten.functional")

from allin1_enhancer import _install_pure_pytorch_natten_fallback
_install_pure_pytorch_natten_fallback()

from natten.functional import natten1dqkrpb as _pt_1d_qkrpb  # type: ignore
from natten.functional import natten1dav as _pt_1d_av  # type: ignore
from natten.functional import natten2dqkrpb as _pt_2d_qkrpb  # type: ignore
from natten.functional import natten2dav as _pt_2d_av  # type: ignore

# Restore real natten
sys.modules["natten"] = _saved_natten
sys.modules["natten.functional"] = _saved_nf

# ──────────────────────────────────────────────────────────────── test helpers

_PASS = 0
_FAIL = 0


def _compare(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    global _PASS, _FAIL
    max_diff = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  [{status}] {name}  (max_diff={max_diff:.2e})")
    if not ok:
        # Show a few mismatched positions for debugging
        diff = (a - b).abs()
        flat = diff.flatten()
        topk_vals, topk_idx = flat.topk(min(5, flat.numel()))
        print(f"         top diffs: {topk_vals.tolist()}")


# ─────────────────────────────────────────────────────────────── 1-D tests

def test_1d_qkrpb(B: int, heads: int, L: int, D: int, ks: int, d: int) -> None:
    tag = f"1d_qkrpb B={B} heads={heads} L={L} D={D} ks={ks} d={d}"
    Q = torch.randn(B, heads, L, D)
    K = torch.randn(B, heads, L, D)
    rpb = torch.randn(heads, 2 * ks - 1)

    ref = _orig_1d_qkrpb(Q, K, rpb, ks, d)
    out = _pt_1d_qkrpb(Q, K, rpb, ks, d)
    _compare(tag, ref, out)


def test_1d_av(B: int, heads: int, L: int, D: int, ks: int, d: int) -> None:
    tag = f"1d_av    B={B} heads={heads} L={L} D={D} ks={ks} d={d}"
    # Build proper attention weights (softmax)
    attn = torch.randn(B, heads, L, ks).softmax(dim=-1)
    V = torch.randn(B, heads, L, D)

    ref = _orig_1d_av(attn, V, ks, d)
    out = _pt_1d_av(attn, V, ks, d)
    _compare(tag, ref, out)


# ─────────────────────────────────────────────────────────────── 2-D tests

def test_2d_qkrpb(B: int, heads: int, H: int, W: int, D: int, ks: int, d: int) -> None:
    tag = f"2d_qkrpb B={B} heads={heads} H={H} W={W} D={D} ks={ks} d={d}"
    Q = torch.randn(B, heads, H, W, D)
    K = torch.randn(B, heads, H, W, D)
    rpb = torch.randn(heads, 2 * ks - 1, 2 * ks - 1)

    ref = _orig_2d_qkrpb(Q, K, rpb, ks, d)
    out = _pt_2d_qkrpb(Q, K, rpb, ks, d)
    _compare(tag, ref, out)


def test_2d_av(B: int, heads: int, H: int, W: int, D: int, ks: int, d: int) -> None:
    tag = f"2d_av    B={B} heads={heads} H={H} W={W} D={D} ks={ks} d={d}"
    attn = torch.randn(B, heads, H, W, ks * ks).softmax(dim=-1)
    V = torch.randn(B, heads, H, W, D)

    ref = _orig_2d_av(attn, V, ks, d)
    out = _pt_2d_av(attn, V, ks, d)
    _compare(tag, ref, out)


# ────────────────────────────────────────────────────────────────── main

def main() -> None:
    torch.manual_seed(42)

    print("=" * 70)
    print("natten native vs pure-PyTorch fallback — A/B comparison")
    print("=" * 70)

    # ── 1-D tests ───────────────────────────────────────────────
    print("\n── 1-D QK+RPB ─────────────────────────────────────────")
    # Actual model params: ks=5, heads=2, head_dim=12
    for d in [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024]:
        L = max(20, 5 * d + 10)
        test_1d_qkrpb(B=1, heads=2, L=L, D=12, ks=5, d=d)

    # Small L (boundary-heavy)
    for L in [5, 6, 7, 8, 10]:
        test_1d_qkrpb(B=2, heads=2, L=L, D=12, ks=5, d=1)

    # Bigger batch
    test_1d_qkrpb(B=4, heads=4, L=100, D=16, ks=5, d=1)
    test_1d_qkrpb(B=4, heads=4, L=100, D=16, ks=5, d=3)

    print("\n── 1-D AV ─────────────────────────────────────────────")
    for d in [1, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256, 512, 1024]:
        L = max(20, 5 * d + 10)
        test_1d_av(B=1, heads=2, L=L, D=12, ks=5, d=d)

    for L in [5, 6, 7, 8, 10]:
        test_1d_av(B=2, heads=2, L=L, D=12, ks=5, d=1)

    # ── 2-D tests ───────────────────────────────────────────────
    print("\n── 2-D QK+RPB ─────────────────────────────────────────")
    for d in [1, 2, 3]:
        H = W = max(10, 5 * d + 3)
        test_2d_qkrpb(B=1, heads=2, H=H, W=W, D=12, ks=5, d=d)

    # Small spatial dims
    test_2d_qkrpb(B=1, heads=2, H=5, W=5, D=12, ks=5, d=1)
    test_2d_qkrpb(B=1, heads=2, H=6, W=8, D=12, ks=5, d=1)

    print("\n── 2-D AV ─────────────────────────────────────────────")
    for d in [1, 2, 3]:
        H = W = max(10, 5 * d + 3)
        test_2d_av(B=1, heads=2, H=H, W=W, D=12, ks=5, d=d)

    test_2d_av(B=1, heads=2, H=5, W=5, D=12, ks=5, d=1)
    test_2d_av(B=1, heads=2, H=6, W=8, D=12, ks=5, d=1)

    # ── summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    total = _PASS + _FAIL
    print(f"Results: {_PASS}/{total} passed, {_FAIL}/{total} failed")
    if _FAIL > 0:
        print("SOME TESTS FAILED!")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")


if __name__ == "__main__":
    main()
