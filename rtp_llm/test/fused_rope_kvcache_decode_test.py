from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

from rtp_llm.ops.fused_rope_kvcache_op import FusedRopeKVCacheDecodeOp


class _SequenceLengths:
    is_cuda = True

    def is_pinned(self) -> bool:
        return False

    def size(self, dim: int) -> int:
        assert dim == 0
        return 2


class _DecodeKernel:
    def __init__(self) -> None:
        self.batch_size = None
        self.kwargs = None

    def decode_fused_rope_kvcache(
        self,
        qkv,
        position_ids,
        batch_size,
        head_num,
        kv_head_num,
        size_per_head,
        kv_cache,
        kv_cache_offset,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.kwargs = kwargs
        return qkv


class FusedRopeKVCacheDecodeOpTest(TestCase):
    def test_forward_matches_decode_kernel_signature(self):
        rope_config = SimpleNamespace(
            style=0,
            dim=128,
            base=10000,
            scale=1,
            factor1=1,
            factor2=1,
            max_pos=8192,
            extrapolation_factor=1,
            mscale=1,
            offset=0,
            index_factor=1,
            mrope_dim1=0,
            mrope_dim2=0,
            mrope_dim3=0,
        )
        attn_configs = SimpleNamespace(
            rope_config=rope_config,
            max_seq_len=8192,
            head_num=16,
            kv_head_num=4,
            size_per_head=128,
            kernel_tokens_per_block=64,
            use_logn_attn=False,
        )
        params = SimpleNamespace(
            position_ids=object(),
            sequence_lengths=_SequenceLengths(),
            kv_cache_offset=object(),
            kv_cache_offset_h=None,
        )
        kv_cache = SimpleNamespace(kv_cache_base=object())
        qkv = object()
        kernel = _DecodeKernel()
        op = FusedRopeKVCacheDecodeOp(attn_configs)

        with (
            patch(
                "rtp_llm.ops.fused_rope_kvcache_op._get_fused_rope_kvcache",
                return_value=kernel,
            ),
            patch(
                "rtp_llm.ops.fused_rope_kvcache_op.get_rope_cache_once",
                return_value=SimpleNamespace(data=None),
            ),
            patch(
                "rtp_llm.ops.fused_rope_kvcache_op.check_rope_cache",
                return_value=False,
            ),
            patch.object(op, "_get_kv_scale", return_value=None),
        ):
            result = op.forward(qkv, kv_cache, params)

        self.assertIs(result, qkv)
        self.assertEqual(kernel.batch_size, 2)
        self.assertEqual(kernel.kwargs["tokens_per_block"], 64)


if __name__ == "__main__":
    main()
