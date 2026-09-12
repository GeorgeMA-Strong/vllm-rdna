# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(
    config_hash="abc123",
    free_memory=78 * GiB_bytes,
    kv_bytes=None,
    gpu_memory_utilization=0.9,
    max_num_seqs=4,
    device_id=0,
):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            compute_hash=lambda: config_hash,
            cache_config=SimpleNamespace(gpu_memory_utilization=gpu_memory_utilization),
            scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        ),
        device=SimpleNamespace(index=device_id),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


@pytest.mark.parametrize(
    "overrides", [{"gpu_memory_utilization": 0.5}, {"max_num_seqs": 16}]
)
def test_startup_plan_reprofiles_when_memory_requirements_change(plan_env, overrides):
    """A graph-cache match must not override a changed memory budget or capacity."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)
    worker = _plan_worker(**overrides)
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes is None


def test_startup_plan_reprofiles_after_rocm_upgrade(plan_env, monkeypatch):
    monkeypatch.setattr(startup_plan.torch.version, "hip", "7.14.0")
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)
    monkeypatch.setattr(startup_plan.torch.version, "hip", "7.15.0")
    worker = _plan_worker()
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes is None


def test_startup_plan_uses_worker_device(plan_env):
    """The same rank can be assigned to a different local device on restart."""
    platform = _plan_platform()
    platform.get_device_total_memory = lambda device_id=0: (
        (80, 32)[device_id] * GiB_bytes
    )
    with patch.object(startup_plan, "current_platform", platform):
        settings = {"free_memory": 28 * GiB_bytes, "gpu_memory_utilization": 0.3}
        maybe_save_startup_plan(_plan_worker(device_id=0, **settings), 20 * GiB_bytes)
        worker = _plan_worker(device_id=1, **settings)
        maybe_apply_startup_plan(worker)
        assert worker.cache_config.kv_cache_memory_bytes is None


@pytest.mark.parametrize("payload", [[], None, True, "broken"])
def test_startup_plan_ignores_non_object_json(plan_env, payload):
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)
    path = next(Path(startup_plan.envs.VLLM_CACHE_ROOT).rglob("startup_plan_*.json"))
    path.write_text(json.dumps(payload))
    worker = _plan_worker()
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes is None


@pytest.mark.parametrize(
    "kv_bytes,baseline",
    [
        (True, 78 * GiB_bytes),
        (1, True),
        (1, -1),
        (0, 78 * GiB_bytes),
        (79 * GiB_bytes, 78 * GiB_bytes),
    ],
)
def test_startup_plan_rejects_invalid_memory_values(plan_env, kv_bytes, baseline):
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)
    path = next(Path(startup_plan.envs.VLLM_CACHE_ROOT).rglob("startup_plan_*.json"))
    payload = json.loads(path.read_text())
    payload.update(kv_cache_memory_bytes=kv_bytes, free_memory_baseline=baseline)
    path.write_text(json.dumps(payload))
    worker = _plan_worker()
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes is None


def test_startup_plan_ignores_invalid_utf8(plan_env):
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)
    path = next(Path(startup_plan.envs.VLLM_CACHE_ROOT).rglob("startup_plan_*.json"))
    path.write_bytes(b"\xff")
    worker = _plan_worker()
    maybe_apply_startup_plan(worker)
    assert worker.cache_config.kv_cache_memory_bytes is None
