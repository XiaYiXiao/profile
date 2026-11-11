"""Opt-in, process-local observation; never edit vLLM files or decisions."""
import os

if os.environ.get("KV_PROBE_DIR"):
    import importlib.abc
    import importlib.machinery
    import sys

    class ProbeLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            return self.original.create_module(spec)

        def exec_module(self, module):
            self.original.exec_module(module)
            from kv_probe import install
            install(module)

    class ProbeFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "vllm.v1.core.sched.scheduler":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.loader:
                spec.loader = ProbeLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, ProbeFinder())
