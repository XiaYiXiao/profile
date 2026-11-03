"""Probe MiMo EP backend availability without loading model weights.

Run with PYTHONPATH=/host/vllm in the same Python environment as serving.
Passing these checks is not an end-to-end model startup or correctness test.
"""

import importlib.metadata
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    determine_expert_placement_strategy,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    select_deepseek_v4_mxfp4_moe_backend,
)
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.models import mimo_v2
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.utils import flashinfer, import_utils


def make_parallel(comm, eplb=False):
    return FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=2, ep_size=4,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
        sp_size=1 if comm == 'flashinfer_nvlink_two_sided' else 2,
        use_ep=True, all2all_backend=comm, enable_eplb=eplb,
    )


def selector_probe(comm, backend, eplb=False):
    config = FusedMoEConfig(
        num_experts=256, experts_per_token=8, hidden_dim=4096,
        intermediate_size=2048, num_local_experts=64, num_logical_experts=256,
        activation=MoEActivation.SILU, device='cuda',
        routing_method=RoutingMethodType.DeepSeekV3,
        moe_parallel_config=make_parallel(comm, eplb),
        in_dtype=torch.bfloat16, router_logits_dtype=torch.float32,
        moe_backend=backend, max_num_tokens=32768,
    )
    result = dict(comm=comm, backend=backend, eplb=eplb)
    try:
        selected, kernel = select_deepseek_v4_mxfp4_moe_backend(config)
        result.update(selected=selected.name, kernel=kernel.__name__)
    except Exception as exc:
        result['error'] = str(exc)
    return result


def model_interface_probe():
    class DummyInner(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.make_empty_intermediate_tensors = lambda *args: None

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(vocab_size=152576, hidden_size=4096)),
        quant_config=None,
    )
    with (
        patch.object(mimo_v2, 'MiMoV2Model', DummyInner),
        patch.object(mimo_v2, 'ParallelLMHead', lambda *a, **kw: torch.nn.Identity()),
        patch.object(mimo_v2, 'LogitsProcessor', lambda *a, **kw: torch.nn.Identity()),
        patch.object(mimo_v2, 'get_pp_group',
                     lambda: SimpleNamespace(is_last_rank=True)),
    ):
        model = mimo_v2.MiMoV2ForCausalLM(vllm_config=config)
    required = ['moe_layers', 'num_moe_layers', 'num_expert_groups',
                'num_logical_experts', 'num_physical_experts',
                'num_local_physical_experts', 'num_routed_experts',
                'num_shared_experts', 'num_redundant_experts', 'expert_weights']
    result = dict(
        scope='Real outer constructor; inner model, LM head and logits processor '
              'stubbed. No model weights or real inference.',
        is_mixture_of_experts=is_mixture_of_experts(model),
        missing_attributes=[k for k in required if not hasattr(model, k)],
    )
    # Exercise the actual validation method without allocating routing tensors.
    try:
        BaseRouter._validate_eplb_state(
            SimpleNamespace(eplb_state=EplbLayerState()))
    except Exception as exc:
        result['uninitialized_router_error'] = str(exc)
    return result


def main():
    result = dict(
        scope='Dependency, kernel-selector, and stubbed model-interface probes only; '
              'no complete server, weights, throughput or multi-GPU communication test.',
        gpu=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        torch=torch.__version__, cuda=torch.version.cuda,
        has_deep_ep=import_utils.has_deep_ep(),
        has_deep_ep_v2=import_utils.has_deep_ep_v2(),
        has_moonep=import_utils.has_moonep(),
        has_humming=import_utils.has_humming(),
        has_flashinfer_nvlink_one_sided=flashinfer.has_flashinfer_nvlink_one_sided(),
        has_flashinfer_nvlink_two_sided=flashinfer.has_flashinfer_nvlink_two_sided(),
        runtime_nccl=import_utils._get_runtime_nccl_version(),
    )
    for package in ['flashinfer-python', 'humming-kernels']:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    from flashinfer.comm.mnnvl import is_multicast_supported
    result['gpu_multicast_supported'] = {
        i: is_multicast_supported(i) for i in range(torch.cuda.device_count())
    }
    comms = ['allgather_reducescatter', 'flashinfer_nvlink_one_sided',
             'flashinfer_nvlink_two_sided', 'deepep_high_throughput',
             'deepep_low_latency']
    backends = ['humming', 'marlin', 'flashinfer_cutlass', 'flashinfer_trtllm', 'triton']
    result['kernel_matrix'] = [selector_probe(c, b) for c in comms for b in backends]
    result['eplb_kernel_only'] = selector_probe('allgather_reducescatter', 'humming', True)
    with patch.dict(os.environ, {'VLLM_HUMMING_MOE_GEMM_TYPE': 'indexed'}):
        result['humming_indexed'] = selector_probe('allgather_reducescatter', 'humming')
    result['round_robin_effective'] = determine_expert_placement_strategy(
        expert_placement_strategy='round_robin',
        moe_parallel_config=make_parallel('allgather_reducescatter'),
        num_expert_group=1, num_redundant_experts=0, enable_eplb=False,
    )
    result['model_eplb_interface'] = model_interface_probe()
    output = Path(__file__).resolve().parents[1] / 'ep_backend_probe.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(output)
    print(json.dumps({k: v for k, v in result.items() if k != 'kernel_matrix'}, indent=2))


if __name__ == '__main__':
    main()
