"""NF4 block dequantization in Triton, a drop-in for bitsandbytes' CUDA kernel.

Every 4-bit projection dequantizes its weight to fp16 twice per step (forward: the
dequant + cuBLAS fallback bitsandbytes takes for M > 512 on a T4; backward: `MatMul4Bit`),
about 390 launches and 5% of the step. This kernel does that unpacking: each byte holds two
NF4 codes (high nibble first), each code selects one of the 16 NormalFloat levels, and each
block of `blocksize` weights is scaled by its fp32 absmax (bitsandbytes dequantizes the
double-quantized absmax itself before calling the backend, so the absmax arrives in fp32 here).

Memory layout is what makes it fast on a T4: one program reads BLOCK contiguous bytes
(vectorized 16-byte loads) and writes BLOCK 32-bit words, each holding the two fp16 values of
one byte (element 0 in the low half, little-endian), so the stores are contiguous and
vectorized too; the level lookup is a tree of selects on the four bits, no table gathers.
Only the fp16 output the training run needs is handled; other dtypes stay with bitsandbytes.

Format and levels: bitsandbytes (Dettmers et al., QLoRA, 2023); the kernel is ours. At patch
time the kernel is checked bit for bit against bitsandbytes' own dequantization on a random
tensor and the patch is skipped if they differ.
"""

import torch
import triton
import triton.language as tl

from .engaged import FELL_BACK, ON, RAN
from .state import DISABLED, fallback

BLOCK = 2048                                                     
NUM_WARPS = 4

                                                                                              
NF4_TABLE = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
             -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
             0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
             0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]


@triton.jit(do_not_specialize=["n_pairs"])
def nf4_dequant_pairs_kernel(packed_ptr, absmax_ptr, out_ptr, n_pairs, QBLOCK: tl.constexpr,
                             BLOCK: tl.constexpr):
    """fp16 output viewed as uint32 pairs: word i = elements (2i, 2i+1) = byte i's nibbles."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    q = tl.load(packed_ptr + offs, mask=mask, other=0).to(tl.int32)
    scale = tl.load(absmax_ptr + (offs * 2) // QBLOCK, mask=mask, other=0.0).to(tl.float32)

                                                                                     
    hi = q >> 4
    b0 = (hi & 1) != 0
    b1 = (hi & 2) != 0
    b2 = (hi & 4) != 0
    b3 = (hi & 8) != 0
    low = tl.where(b2, tl.where(b1, tl.where(b0, 0.0, -0.09105003625154495),
                                tl.where(b0, -0.18477343022823334, -0.28444138169288635)),
                   tl.where(b1, tl.where(b0, -0.39491748809814453, -0.5250730514526367),
                            tl.where(b0, -0.6961928009986877, -1.0)))
    high = tl.where(b2, tl.where(b1, tl.where(b0, 1.0, 0.7229568362236023),
                                 tl.where(b0, 0.5626170039176941, 0.44070982933044434)),
                    tl.where(b1, tl.where(b0, 0.33791524171829224, 0.24611230194568634),
                             tl.where(b0, 0.16093020141124725, 0.07958029955625534)))
    first = (tl.where(b3, high, low) * scale).to(tl.float16).to(tl.uint16, bitcast=True)

    lo = q & 15
    b0 = (lo & 1) != 0
    b1 = (lo & 2) != 0
    b2 = (lo & 4) != 0
    b3 = (lo & 8) != 0
    low = tl.where(b2, tl.where(b1, tl.where(b0, 0.0, -0.09105003625154495),
                                tl.where(b0, -0.18477343022823334, -0.28444138169288635)),
                   tl.where(b1, tl.where(b0, -0.39491748809814453, -0.5250730514526367),
                            tl.where(b0, -0.6961928009986877, -1.0)))
    high = tl.where(b2, tl.where(b1, tl.where(b0, 1.0, 0.7229568362236023),
                                 tl.where(b0, 0.5626170039176941, 0.44070982933044434)),
                    tl.where(b1, tl.where(b0, 0.33791524171829224, 0.24611230194568634),
                             tl.where(b0, 0.16093020141124725, 0.07958029955625534)))
    second = (tl.where(b3, high, low) * scale).to(tl.float16).to(tl.uint16, bitcast=True)

    word = first.to(tl.uint32) | (second.to(tl.uint32) << 16)
    tl.store(out_ptr + offs, word, mask=mask)


def nf4_dequantize(packed, absmax, blocksize, out):
    """packed: uint8 [(n+1)//2, 1] or flat; absmax: fp32 [ceil(n / blocksize)]; out: contiguous
    fp16 tensor with an even number of elements."""
    packed = packed.contiguous().view(-1)
    n = out.numel()
    words = out.view(-1).view(torch.int32)
    nf4_dequant_pairs_kernel[(triton.cdiv(n // 2, BLOCK),)](
        packed, absmax, words, n // 2, QBLOCK=blocksize, BLOCK=BLOCK, num_warps=NUM_WARPS)
    return out


def _self_check(original_impl):
    """Our kernel against bitsandbytes' dequantization on a random NF4 tensor: exact match."""
    import bitsandbytes.functional as bnbf

    generator = torch.Generator(device="cuda").manual_seed(0)                              
    w = torch.randn(1024, 1536, device="cuda", dtype=torch.float16, generator=generator)
    packed, state = bnbf.quantize_4bit(w, blocksize=64, quant_type="nf4", compress_statistics=True)
    absmax = bnbf.dequantize_blockwise(state.absmax, state.state2) + state.offset
    ref = torch.empty(w.shape, dtype=torch.float16, device="cuda")
    original_impl(packed, absmax, 64, "nf4", torch.float16, out=ref)
    ours = torch.empty_like(ref)
    nf4_dequantize(packed, absmax, 64, ours)
    torch.cuda.synchronize()
    return torch.equal(ours, ref)


def patch_bnb_dequant(verify=True):
    """Route `bitsandbytes.backends.cuda.ops._dequantize_4bit_impl` (both the forward fallback
    and `MatMul4Bit.backward` reach it) through the Triton kernel for NF4 fp16 on CUDA."""
    try:
        import bitsandbytes.backends.cuda.ops as ops
    except Exception:                                                         
        return False
    original = ops._dequantize_4bit_impl
    if getattr(original, "_solution_patched", False):
        return True
    if verify:
        if not torch.cuda.is_available():
            return False
        try:
            if not _self_check(original):
                return False
        except Exception:                                                                    
            return False

    def impl(A, absmax, blocksize, quant_type, dtype, out):
        if (quant_type != "nf4" or not A.is_cuda or absmax.dtype != torch.float32
                or blocksize % 2 or dtype != torch.float16 or out.dtype != torch.float16
                or not out.is_contiguous() or out.numel() % 2 or "nf4_dequant" in DISABLED):
            FELL_BACK["nf4_dequant"] += 1
            return original(A, absmax, blocksize, quant_type, dtype, out=out)
        try:                                                                                 
            nf4_dequantize(A, absmax, blocksize, out)
        except Exception as exc:                               
            if not fallback("nf4_dequant", exc):
                raise
        else:
            RAN["nf4_dequant"] += 1
            return None
        FELL_BACK["nf4_dequant"] += 1
        return original(A, absmax, blocksize, quant_type, dtype, out=out)

    impl._solution_patched = True
    ops._dequantize_4bit_impl = impl
    ON.add("nf4_dequant")
    return True
