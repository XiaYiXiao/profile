"""Resolve the scripts' EPLB environment settings without loading vLLM/GPU code."""

import json
import os
import sys


def resolve():
    defaults = {
        "window_size": "8",
        "step_interval": "16",
        "num_redundant_experts": "0",
        "use_async": "true",
        "communicator": "",
        "log_balancedness": "true",
        "log_balancedness_interval": "1",
    }
    config = {
        key: os.environ.get("EPLB_" + key.upper(), default)
        for key, default in defaults.items()
    }
    if raw := os.environ.get("EPLB_CONFIG", "").strip():
        overrides = json.loads(raw)
        if not isinstance(overrides, dict):
            raise ValueError("EPLB_CONFIG 必须是 JSON 对象")
        config.update(overrides)
    for key in [
        "window_size",
        "step_interval",
        "num_redundant_experts",
        "log_balancedness_interval",
    ]:
        value = int(str(config[key]))
        minimum = 0 if key == "num_redundant_experts" else 1
        if value < minimum:
            raise ValueError(f"{key} 必须是 >= {minimum} 的整数")
        config[key] = value
    for key in ["use_async", "log_balancedness"]:
        value = str(config[key]).lower()
        if value not in ["true", "false", "0", "1"]:
            raise ValueError(f"{key} 必须为 true/false 或 1/0")
        config[key] = value in ["true", "1"]
    config["communicator"] = config["communicator"] or None
    if config["communicator"] not in [
        None,
        "torch_gloo",
        "torch_nccl",
        "pynccl",
        "nixl",
        "torch_xccl",
    ]:
        raise ValueError("EPLB_COMMUNICATOR 不受当前 vLLM 支持")
    if config["use_async"] and config["communicator"] in ["pynccl", "torch_nccl"]:
        raise ValueError(
            "异步 EPLB 不支持 pynccl/torch_nccl；请设置 EPLB_USE_ASYNC=false"
        )
    return config


if __name__ == "__main__":
    try:
        print(json.dumps(resolve(), separators=(",", ":")))
    except (ValueError, TypeError) as error:
        sys.exit(f"EPLB 配置错误：{error}")
