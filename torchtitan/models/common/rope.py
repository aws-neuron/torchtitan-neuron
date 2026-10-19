# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Modifications Copyright (c) Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the BSD-3-Clause license.

# Shape suffixes: B = batch, L = sequence, N = heads, H = head dimension,
# C = cache width (H / 2 for complex caches, 2 * H for cos/sin caches).

import math
from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.fx.experimental.symbolic_shapes import guard_or_false

from torchtitan.protocols.module import Module

__all__ = [
    "ComplexRoPE",
    "CosSinRoPE",
    "RoPE",
]


@spmd.no_typecheck()
def _maybe_check_max_pos(positions: torch.Tensor, *, max_valid_pos: int) -> None:
    """Async bounds check: verify all position values <= max_valid_pos.

    Uses ``torch._assert_async`` to avoid a device-host sync while still
    catching out-of-bounds positions (the assertion failure surfaces at a
    later kernel launch).  Skipped entirely under ``torch.compile``.
    """
    if torch.compiler.is_compiling():
        return
    pos_local = positions.to_local() if isinstance(positions, DTensor) else positions
    torch._assert_async(
        torch.all(pos_local <= max_valid_pos),
        f"position_ids exceed {max_valid_pos=}",
    )


class RoPE(Module):
    """Shared Rotary Position Embedding module.

    Common base for concrete RoPE formats. Use ``ComplexRoPE.Config`` for
    complex exponential caches and ``CosSinRoPE.Config`` for concatenated
    cosine/sine caches.
    """

    _cp_rank: int = 0
    _cp_size: int = 1
    _cp_load_balancer: str | None = None

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        max_seq_len: int
        theta: float = 10000.0
        scaling: Literal["none", "llama", "yarn"] = "none"
        # llama scaling params
        scaling_factor: float = 8.0
        low_freq_factor: float = 1.0
        high_freq_factor: float = 4.0
        original_max_position_embeddings: int = 8192
        # yarn scaling params
        rope_factor: float = 1.0
        beta_fast: float = 32.0
        beta_slow: float = 1.0
        original_seq_len: int = 4096
        mscale: float = 0.0
        use_prefix_cache_for_positions: bool = False
        # For causal, non-packed training with sequential position IDs, build
        # the cache from the sequence prefix instead of gathering cache rows.

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.register_buffer("cache", self._precompute_cache(), persistent=False)

    @property
    def uses_prefix_cache_for_positions(self) -> bool:
        return self.config.use_prefix_cache_for_positions

    def _precompute_cache(self) -> torch.Tensor:
        """Build the reusable cache for all positions up to ``max_seq_len``.

        Returns:
            RoPE cache for all valid positions.
        """
        raise NotImplementedError

    def _reshape_prefix_cache(self, query_BLNH: torch.Tensor) -> torch.Tensor:
        """Select sequential cache rows for this rank's context-parallel shard."""
        if self._cp_size == 1:
            return _reshape_for_broadcast(self.cache, query_BLNH.shape, None)
        if isinstance(query_BLNH, DTensor):
            if not isinstance(self.cache, DTensor) or not all(
                isinstance(p, Replicate) for p in self.cache.placements
            ):
                raise ValueError(
                    "DTensor queries require a replicated DTensor RoPE cache"
                )
            # The cache is broadcast over batch and heads, and only its
            # sequence dimension follows the query's sharding.
            placements = tuple(
                p if isinstance(p, Shard) and p.dim == 1 else Replicate()
                for p in query_BLNH.placements
            )
            cache_1L1C = _reshape_prefix_cache(
                self.cache.to_local(),
                query_BLNH.to_local(),
                self._cp_rank,
                self._cp_size,
                self._cp_load_balancer,
            )
            return DTensor.from_local(
                cache_1L1C, query_BLNH.device_mesh, placements, run_check=False
            )
        return _reshape_prefix_cache(
            self.cache,
            query_BLNH,
            self._cp_rank,
            self._cp_size,
            self._cp_load_balancer,
        )

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a cache aligned to ``query`` and ``positions``.

        Args:
            query: Query tensor of shape ``(batch, seq_len, n_heads, head_dim)``.
            positions: Optional position IDs of shape ``(batch, seq_len)``.

        Returns:
            Prepared RoPE cache for the concrete RoPE format.
        """
        raise NotImplementedError

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a prepared RoPE cache to query and key.

        Args:
            query: Query tensor of shape ``(batch, seq_len, n_heads, head_dim)``.
            key: Key tensor of shape ``(batch, seq_len, n_heads, head_dim)``.
            rope_cache: Prepared cache broadcastable to ``query`` and ``key``
                according to the concrete RoPE format.

        Returns:
            Rotated query and key tensors with the same shapes and dtypes as
            ``query`` and ``key``.
        """
        raise NotImplementedError

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors."""
        reshaped_cache = self._reshape_cache(query, positions)
        return self.apply_rotary_emb(query, key, reshaped_cache)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # TODO: In long-term we need to have buffer abstraction in `Module`` class to infer the buffer_device
        if buffer_device is None:
            # After ``to_empty()``, the existing cache records the target device.
            # Recompute there when the caller does not pass an explicit buffer device.
            buffer_device = self.cache.device
        with torch.device(buffer_device):
            self.cache = self._precompute_cache()


@dataclass(kw_only=True, slots=True)
class ComplexRoPEConfig(RoPE.Config):
    pass


class ComplexRoPE(RoPE):
    # Keep the public ``ComplexRoPE.Config`` API, while giving torch.compile's
    # caching precompile guard serializer a top-level dataclass type.
    Config = ComplexRoPEConfig

    def _precompute_cache(self) -> torch.Tensor:
        """Precompute complex cis values.

        Returns:
            Cache of shape ``(max_seq_len, dim / 2)``.
        """
        cfg = self.config
        dim = cfg.dim
        end = cfg.max_seq_len
        theta = cfg.theta

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

        if cfg.scaling == "llama":
            scaling_factor = cfg.scaling_factor
            low_freq_factor = cfg.low_freq_factor
            high_freq_factor = cfg.high_freq_factor
            original_max_position_embeddings = cfg.original_max_position_embeddings
            wavelen = 2 * math.pi / freqs
            high_freq_wavelen = original_max_position_embeddings / high_freq_factor
            low_freq_wavelen = original_max_position_embeddings / low_freq_factor
            freqs = torch.where(
                wavelen > low_freq_wavelen, freqs / scaling_factor, freqs
            )
            smooth_factor = (
                original_max_position_embeddings / wavelen - low_freq_factor
            ) / (high_freq_factor - low_freq_factor)
            smoothed_freqs = (
                1 - smooth_factor
            ) * freqs / scaling_factor + smooth_factor * freqs
            is_medium_freqs = ~(wavelen < high_freq_wavelen) * ~(
                wavelen > low_freq_wavelen
            )
            freqs = torch.where(is_medium_freqs, smoothed_freqs, freqs)
        elif cfg.scaling == "yarn" and end > cfg.original_seq_len:
            # YaRN (DeepSeek V3 style)
            beta_fast = cfg.beta_fast
            beta_slow = cfg.beta_slow
            base = theta
            original_seq_len = cfg.original_seq_len
            factor = cfg.rope_factor

            def find_correction_dim(
                num_rotations: float, dim: int, base: float, max_seq_len: int
            ) -> float:
                return (
                    dim
                    * math.log(max_seq_len / (num_rotations * 2 * math.pi))
                    / (2 * math.log(base))
                )

            def find_correction_range(
                low_rot: float,
                high_rot: float,
                dim: int,
                base: float,
                max_seq_len: int,
            ) -> tuple[int, int]:
                low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
                high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
                return max(low, 0), min(high, dim - 1)

            def linear_ramp_factor(
                min_val: float, max_val: float, dim: int
            ) -> torch.Tensor:
                if min_val == max_val:
                    max_val += 0.001
                linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
                    max_val - min_val
                )
                return torch.clamp(linear_func, 0, 1)

            low, high = find_correction_range(
                beta_fast, beta_slow, dim, base, original_seq_len
            )
            smooth = 1 - linear_ramp_factor(low, high, dim // 2)
            freqs = freqs / factor * (1 - smooth) + freqs * smooth

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return complex cache shaped for query/key broadcast.

        Returns:
            Cache of shape ``(1 or batch, seq_len, 1, dim / 2)``.
        """
        # Complex RoPE cache has width dim / 2 because each complex value
        # represents a pair of real dimensions.
        complex_query_shape = (*query.shape[:-1], query.shape[-1] // 2)
        if self.uses_prefix_cache_for_positions:
            return self._reshape_prefix_cache(query)
        positions = _maybe_wrap_positions(positions, query)
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        return _reshape_for_broadcast(self.cache, complex_query_shape, positions)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply complex RoPE using adjacent-dim pairs."""
        xq_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        xq_out = torch.view_as_real(xq_ * rope_cache).flatten(3)
        xk_out = torch.view_as_real(xk_ * rope_cache).flatten(3)
        return xq_out.type_as(query), xk_out.type_as(key)


@dataclass(kw_only=True, slots=True)
class CosSinRoPEConfig(RoPE.Config):
    pass


class CosSinRoPE(RoPE):
    # Keep the public ``CosSinRoPE.Config`` API, while giving torch.compile's
    # caching precompile guard serializer a top-level dataclass type.
    Config = CosSinRoPEConfig

    def _precompute_cache(self) -> torch.Tensor:
        """Precompute cos/sin values.

        Returns:
            Cache of shape ``(max_seq_len, dim * 2)``.
        """
        cfg = self.config
        dim = cfg.dim
        max_seq_len = cfg.max_seq_len
        base = cfg.theta

        freq = base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        mscale = 1.0

        if cfg.scaling == "llama":
            raise NotImplementedError("Cos/sin RoPE does not support Llama scaling.")

        if cfg.scaling == "yarn" and cfg.rope_factor > 1.0:
            rope_factor = cfg.rope_factor
            # YaRN mscale for attention magnitude preservation
            mscale = 0.1 * math.log(rope_factor) + 1.0

            # Compute correction range (NTK by parts)
            d_half = dim / 2
            low = (
                d_half
                * math.log(cfg.original_seq_len / (cfg.beta_slow * 2 * math.pi))
                / math.log(base)
            )
            high = (
                d_half
                * math.log(cfg.original_seq_len / (cfg.beta_fast * 2 * math.pi))
                / math.log(base)
            )
            assert (
                0 < low < high < d_half - 1
            ), f"Invalid YaRN params: 0 < {low} < {high} < {d_half - 1}"

            ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            interpolation = 1.0 / (rope_factor * freq)
            extrapolation = 1.0 / freq
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            inv_freq = 1.0 / freq

        t = torch.arange(max_seq_len, dtype=inv_freq.dtype, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq).float()
        theta = torch.cat([freqs, freqs], dim=-1)

        cos = theta.cos() * mscale
        sin = theta.sin() * mscale
        return torch.cat([cos, sin], dim=-1)

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return cos/sin cache shaped for query/key broadcast.

        Returns:
            Cache of shape ``(1 or batch, seq_len, 1, dim * 2)``.
        """
        if self.uses_prefix_cache_for_positions:
            return self._reshape_prefix_cache(query)
        positions = _maybe_wrap_positions(positions, query)
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        return _reshape_for_broadcast(self.cache, query.shape, positions)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cos/sin RoPE using the rotate-half convention."""
        head_dim = query.shape[-1]
        cos = rope_cache[..., :head_dim]
        sin = rope_cache[..., head_dim:]
        query_f = query.float()
        key_f = key.float()
        xq_out = (query_f * cos) + (CosSinRoPE._rotate_half(query_f) * sin)
        xk_out = (key_f * cos) + (CosSinRoPE._rotate_half(key_f) * sin)
        return xq_out.type_as(query), xk_out.type_as(key)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


@spmd.local_map(out_types={"dp": spmd.R, "cp": spmd.S(1), "tp": spmd.R})
def _reshape_prefix_cache(
    cache_LC: torch.Tensor,
    query_BLNH: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    load_balancer: str | None,
) -> torch.Tensor:
    """Build a broadcast cache without gathering device-side position IDs."""
    seq_len = query_BLNH.shape[1]
    if seq_len * cp_size > cache_LC.shape[0]:
        raise ValueError("The context-parallel sequence exceeds the RoPE cache")
    if load_balancer is None:
        start = cp_rank * seq_len
        cache_LC = cache_LC[start : start + seq_len]
    elif load_balancer == "headtail":
        if seq_len % 2:
            raise ValueError(
                "Head-tail context parallelism requires even shard lengths"
            )
        half = seq_len // 2
        head = cp_rank * half
        tail = (2 * cp_size - cp_rank - 1) * half
        cache_LC = torch.cat(
            (cache_LC[head : head + half], cache_LC[tail : tail + half])
        )
    else:
        raise ValueError(
            "Prefix RoPE caches require context_parallel_load_balancer=None "
            "or 'headtail'"
        )
    return cache_LC.view(1, seq_len, 1, cache_LC.shape[-1])


@spmd.local_map(out_types={"dp": spmd.S(0), "cp": spmd.S(1), "tp": spmd.R})
def _reshape_for_broadcast(
    rope_cache: torch.Tensor,
    query_shape: torch.Size | tuple[int, ...],
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reshape a RoPE cache for broadcasting with query/key tensors."""
    ndim = len(query_shape)
    assert ndim > 1
    bsz, seqlen = query_shape[:2]
    # cache_width is `head_dim * 2` for CosSinRoPE, and `head_dim // 2` for ComplexRoPE
    cache_width = rope_cache.shape[-1]
    if positions is None:
        # No explicit positions: use the prefix cache and broadcast it over batch.
        rope_cache = rope_cache[0:seqlen]
        # assert rope_cache.shape == (seqlen, cache_width)
        shape = [
            d if i == 1 else cache_width if i == ndim - 1 else 1
            for i, d in enumerate(query_shape)
        ]
        return rope_cache.view(*shape)

    # TODO(pianpwk): Remove this vLLM inference compatibility branch once
    # singleton positions can use the general gather path; see PR #3750.
    # Concrete/provable singleton positions can use the cheaper prefix-shaped
    # view path. If singleton-ness is symbolic, fall through to gather below.
    if guard_or_false(positions.size(0) == 1):
        # assert positions.shape == (1, seqlen)
        rope_cache = rope_cache[positions.squeeze(0)]
        # assert rope_cache.shape == (seqlen, cache_width)
        shape = [
            d if i == 1 else cache_width if i == ndim - 1 else 1
            for i, d in enumerate(query_shape)
        ]
        return rope_cache.view(*shape)
    else:
        # Per-batch positions, plus singleton positions whose first dimension was
        # not statically provable above, use the general gather path.
        # assert positions.shape == (bsz, seqlen)
        positions = positions.expand(bsz, -1)
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bsz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bsz, seqlen, 1, 1).expand(bsz, seqlen, 1, cache_width),
        )
        return rope_cache


def _maybe_wrap_positions(
    positions: torch.Tensor | None,
    x: torch.Tensor,
) -> torch.Tensor | None:
    """Wrap positions as a DTensor deriving mesh and placements from x (xq/xk).

    TODO: In a full DTensor rewrite, positions should be made a DTensor
    in/right after dataloading, together with inputs and labels.

    When TP uses use_local_output=False (DeepSeek V3, Qwen3, GPT-OSS),
    x is a DTensor but positions is a plain tensor. The downstream
    torch.gather requires both operands to be the same type.

    Positions (bsz, seqlen) has fewer dimensions than x (bsz, seqlen,
    n_heads, head_dim), so we only preserve Shard placements for shared
    dimensions. Shard dims beyond positions' rank (e.g. Shard(2) for TP
    on heads) become Replicate.
    """
    if (
        positions is not None
        and isinstance(x, DTensor)
        and not isinstance(positions, DTensor)
    ):
        ndim = positions.ndim
        placements = tuple(
            p if not isinstance(p, Shard) or p.dim < ndim else Replicate()
            for p in x.placements
        )
        positions = DTensor.from_local(
            positions,
            x.device_mesh,
            placements,
            run_check=False,
        )
    return positions
