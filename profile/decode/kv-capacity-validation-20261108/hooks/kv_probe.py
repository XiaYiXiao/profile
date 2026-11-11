"""Observe existing allocator returns and scheduler output, without changing them.

All wrapped methods are called exactly once with their original arguments.
No cache allocation is retried or simulated by the probe. Disk records are
buffered; there is one file per EngineCore process. Timing has probe overhead.
"""
import functools
import json
import os
from pathlib import Path
import time


def install(module):
    cls = module.Scheduler
    if getattr(cls, "_kv_probe_installed", False):
        return
    cls._kv_probe_installed = True
    original_init = cls.__init__
    original_schedule = cls.schedule

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        rank = self.parallel_config.data_parallel_rank
        directory = Path(os.environ["KV_PROBE_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        stream = (directory / f"engine-dp{rank}-pid{os.getpid()}.jsonl").open("a")
        state = {"step": 0, "failures": 0, "checks": [], "last_flush": 0.0}
        self._kv_probe_state = state
        manager = self.kv_cache_manager
        pool = manager.block_pool

        def emit(kind, **fields):
            stream.write(json.dumps(dict(
                kind=kind, time_unix_s=time.time(), dp_rank=rank,
                pid=os.getpid(), step=state["step"], **fields,
            ), separators=(",", ":")) + "\n")
            now = time.monotonic()
            if now - state["last_flush"] > 1:
                stream.flush()
                state["last_flush"] = now

        self._kv_probe_emit = emit
        self._kv_probe_stream = stream
        groups = []
        for group in self.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            groups.append(dict(
                type=type(spec).__name__, layers=len(group.layer_names),
                block_size=spec.block_size, page_bytes=spec.page_size_bytes,
                sliding_window=getattr(spec, "sliding_window", None),
            ))
        emit("configuration", total_blocks=self.kv_cache_config.num_blocks,
             initial_free_blocks=pool.get_num_free_blocks(), groups=groups,
             max_num_seqs=self.max_num_running_reqs,
             max_active_seqs=self.max_num_active_reqs,
             max_scheduled_tokens=self.max_num_scheduled_tokens,
             max_batched_tokens=self.scheduler_config.max_num_batched_tokens,
             max_model_len=self.max_model_len,
             watermark_blocks=manager.watermark_blocks,
             reserve_full_isl=self.scheduler_reserve_full_isl,
             async_scheduling=self.scheduler_config.async_scheduling)
        stream.flush()

        original_count = manager.coordinator.get_num_blocks_to_allocate

        @functools.wraps(original_count)
        def count(*a, **kw):
            value = original_count(*a, **kw)
            state["checks"].append(dict(
                required_blocks=value, free_blocks=pool.get_num_free_blocks(),
                admission_check=kw.get("apply_admission_cap", False),
                tokens_need_slot=kw.get("num_tokens"),
                total_computed_tokens=kw.get("total_computed_tokens"),
            ))
            return value

        manager.coordinator.get_num_blocks_to_allocate = count
        original_allocate = manager.allocate_slots

        @functools.wraps(original_allocate)
        def allocate(request, num_new_tokens, *a, **kw):
            state["checks"] = []
            free_before = pool.get_num_free_blocks()
            result = original_allocate(request, num_new_tokens, *a, **kw)
            if result is None:
                state["failures"] += 1
                emit("allocation_failed", request_id=request.request_id,
                     request_status=request.status.name,
                     prompt_tokens=request.num_prompt_tokens,
                     computed_tokens=request.num_computed_tokens,
                     request_tokens=request.num_tokens,
                     new_tokens=num_new_tokens,
                     external_tokens=kw.get("num_external_computed_tokens", 0),
                     lookahead_tokens=kw.get("num_lookahead_tokens", 0),
                     reserved_blocks=kw.get("reserved_blocks", 0),
                     free_before=free_before, free_after=pool.get_num_free_blocks(),
                     checks=state["checks"])
            return result

        manager.allocate_slots = allocate

    @functools.wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        state = self._kv_probe_state
        state["step"] += 1
        state["failures"] = 0
        t0 = time.monotonic()
        result = original_schedule(self, *args, **kwargs)
        free = self.kv_cache_manager.block_pool.get_num_free_blocks()
        total = self.kv_cache_config.num_blocks
        self._kv_probe_emit(
            "step", scheduled_requests=len(result.num_scheduled_tokens),
            scheduled_tokens=result.total_num_scheduled_tokens,
            running=len(self.running), waiting=len(self.waiting),
            kv_holding_waiting=len(self.kv_holding_waiting),
            total_blocks=total, free_blocks=free, kv_usage=1-free/total,
            allocation_failures=state["failures"],
            scheduler_wall_s=time.monotonic()-t0,
        )
        return result

    cls.__init__ = init
    cls.schedule = schedule
