import torch
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from loguru import logger
from flash_attn.flash_attn_interface import flash_attn_func

from .int_flashattention.flash_atten_int8 import attention_int8 as _attention_int8


def attention_naive(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

def attention_replicate(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    # Only enable flash attention backend
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def quant_pertoken(X):
    X_max, _ = torch.abs(X).max(dim=-1)
    X_scale = X_max / 127
    ret = torch.round(X / X_scale[:, :, :, None]).to(torch.int8)
    return ret, X_scale

def quant_pertensor(X):
    X_max, _ = torch.abs(X).max(dim=-1)
    X_max, _ = torch.max(X_max, dim=-1)
    X_scale = X_max / 127
    ret = torch.round(X / X_scale[:, :, None, None]).to(torch.int8)
    return ret, X_scale


def attention_int8(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    q8, qs8 = quant_pertoken(q)
    k8, ks8 = quant_pertoken(k)
    # v8, vs8 = quant_pertensor(v)

    x = _attention_int8(q8, k8, v, qs8, ks8, causal=False, scale=1.)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def attention_fa3(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)
    q = q.to(torch.float8_e4m3fn)
    k = k.to(torch.float8_e4m3fn)
    v = v.to(torch.float8_e4m3fn)

    x = flash_attn_func(q, k, v, causal=False)
    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


attention_mode = "replicate"

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    if attention_mode == "replicate":
        return attention_replicate(q, k, v, pe)

    elif attention_mode == "naive":
        return attention_naive(q, k, v, pe)

    elif attention_mode == "int8":
        return attention_int8(q, k, v, pe)

    elif attention_mode == "fa3":
        return attention_fa3(q, k, v, pe)

    else:
        raise NotImplementedError

def set_attention_mode(mode):
    assert mode in ["naive", "replicate", "int8", "fa3"]

    global attention_mode
    attention_mode = mode
