# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-based one-shot all-reduce for small TP messages on gfx1030 (2..8 ranks).

vLLM's XGMI pull custom all-reduce is the wrong kernel on RDNA PCIe (device
flags are not visible across the link; peer STORE beats peer LOAD). This
path pushes into uncached staging, signals on a host-coherent flag page, and
reduces in fixed rank order. Graph-capture safe (sequence numbers live on
device, not in kernel arguments).

Default is off. Enable with VLLM_RDNA_AR=1, VLLM_RDNA_AR=auto (PIX mesh:
amdsmi PCIE hops<=2, one switch), or VLLM_FORCE_CUSTOM_ALL_REDUCE on
gfx10x. VLLM_RDNA_AR=0 always disables. VLLM_RDNA_AR_BLOCKS /
VLLM_RDNA_AR_PACE pace PCIe push bursts; VLLM_RDNA_AR_MAX_KB (default 512)
bounds the fast path.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.distributed.device_communicators.rdna_p2p import (
    p2p_level_from_env,
    should_init_rdna_ar,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_instances = 0


class RdnaOneShotAllReduce:
    def __init__(self, group: ProcessGroup, device: torch.device) -> None:
        global _instances
        from vllm import _custom_ops as ops

        self.disabled = True
        self.handle = -1
        self._ops = ops
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        max_kb = int(os.getenv("VLLM_RDNA_AR_MAX_KB", "512"))
        self.max_bytes = max_kb * 1024
        if not (2 <= self.world_size <= 8):
            return
        dev_idx = (
            device.index
            if device.index is not None
            else torch.accelerator.current_device_index()
        )
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, int(dev_idx), group=group)
        pix = False
        try:
            physical = [
                current_platform.visible_device_id_to_physical_device_id(int(d))
                for d in gathered
            ]
            pix = bool(current_platform.is_pix_connected(physical))
        except Exception as e:  # noqa: BLE001
            logger.warning("rdna_ar: PIX topology query failed (%s)", e)
        pix_votes: list = [None] * self.world_size
        dist.all_gather_object(pix_votes, pix, group=group)
        pix = all(bool(v) for v in pix_votes)
        if not should_init_rdna_ar(
            os.environ.get("VLLM_RDNA_AR"),
            envs.VLLM_FORCE_CUSTOM_ALL_REDUCE,
            pix,
        ):
            logger.info(
                "rdna_ar: skipped (VLLM_RDNA_AR=%s force=%s pix=%s "
                "level=%s devices=%s)",
                os.environ.get("VLLM_RDNA_AR"),
                envs.VLLM_FORCE_CUSTOM_ALL_REDUCE,
                pix,
                p2p_level_from_env(),
                gathered,
            )
            return
        device_ids = torch.tensor(gathered, dtype=torch.int64)
        # rank 0 names the flag page; one per instance
        my_name = f"/vllm_rdna_ar_{os.getpid()}_{_instances}"
        names: list = [None] * self.world_size
        dist.all_gather_object(names, my_name, group=group)
        shm_name = names[0]
        _instances += 1

        # ordered init: rank 0 (re)creates the flag page before anyone opens it.
        # Every rank executes every barrier no matter what happens locally.
        packed = None
        err: str | None = None
        for r in range(self.world_size):
            if r == self.rank and err is None:
                try:
                    with torch.accelerator.device_index(dev_idx):
                        packed = ops.rdna_ar_init(
                            self.rank,
                            self.world_size,
                            device_ids,
                            self.max_bytes,
                            shm_name,
                        )
                except Exception as e:  # noqa: BLE001
                    err = str(e)
            dist.barrier(group=group)
        status: list = [None] * self.world_size
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            logger.warning(
                "rdna_ar: disabled for this group -- init failed on some rank: %s",
                [s for s in status if s is not None][:1],
            )
            return
        assert packed is not None
        raw = packed.numpy().tobytes()
        self.handle = int.from_bytes(raw[:8], "little", signed=True)
        handles: list = [None] * self.world_size
        dist.all_gather_object(handles, raw[8:], group=group)
        buf = torch.frombuffer(bytearray(b"".join(handles)), dtype=torch.uint8).view(
            self.world_size, -1
        )
        err = None
        try:
            with torch.accelerator.device_index(dev_idx):
                ops.rdna_ar_connect(self.handle, buf.contiguous())
        except Exception as e:  # noqa: BLE001
            err = str(e)
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            logger.warning("rdna_ar: disabled -- connect failed: %s", status)
            return
        dist.barrier(group=group)

        # Boot self-test (2026-08-31): on boards where GPU P2P is slow or broken
        # (ACS redirect, chipset-routed slots, cross-socket paths) init can succeed
        # while every collective then spins to its ~2 s cap and aborts WITHOUT
        # writing the output -- silent corruption plus stalls that get reported by
        # whatever waits next (usually the PLE handshake). Verify the path with
        # known patterns before trusting it; all ranks agree on the verdict.
        err = self._self_test(device, group)
        dist.all_gather_object(status, err, group=group)
        if any(s is not None for s in status):
            logger.warning(
                "rdna_ar: disabled -- boot self-test failed on some rank "
                "(weak GPU peer-to-peer on this board? falling back to RCCL): %s",
                [s for s in status if s is not None],
            )
            return
        dist.barrier(group=group)
        self.disabled = False
        logger.info(
            "rdna_ar: one-shot all-reduce active (handle %d, rank %d/%d, "
            "devices %s, pix=%s, max %d KB; blocks cap %s, pace %s)",
            self.handle,
            self.rank,
            self.world_size,
            gathered,
            pix,
            max_kb,
            os.getenv("VLLM_RDNA_AR_BLOCKS", "auto"),
            os.getenv("VLLM_RDNA_AR_PACE", "0"),
        )

    def _self_test(self, device: torch.device, group: ProcessGroup) -> str | None:
        """Verify the fast path at three sizes.

        Returns an error string or None. Per size: one untimed warm-up
        (loads the code object and first-touches IPC), then REPEATS timed
        collectives judged on their minimum. A slow P2P path cannot hide
        that minimum; the max only measures how far ranks were out of step
        at boot. Every collective checks the spin-cap and the fp32 result.
        """
        import time

        debug = os.environ.get("VLLM_RDNA_AR_DEBUG") == "1"
        dev_idx = (
            device.index
            if device.index is not None
            else torch.accelerator.current_device_index()
        )
        REPEATS = 3
        err: str | None = None
        try:
            with torch.accelerator.device_index(dev_idx):
                for trial, numel in enumerate((1024, 4096, self.max_bytes // 2)):
                    inp = torch.full(
                        (numel,),
                        float(self.rank + 1) * (trial + 1),
                        dtype=torch.float16,
                        device=device,
                    )
                    expect = float(
                        (trial + 1) * self.world_size * (self.world_size + 1) // 2
                    )
                    times: list[float] = []
                    for rep in range(REPEATS + 1):  # rep 0 = warm-up, untimed
                        # Barriers bound cross-rank skew before the ~2 s spin cap.
                        # Warm-up covers the per-rank cold start.
                        dist.barrier(group=group)
                        if debug:
                            print(
                                f"[rdna_ar rank{self.rank}] trial{trial} "
                                f"rep{rep} barrier-out",
                                flush=True,
                            )
                        # Never leave the loop early: a rank that stops calling barriers
                        # while its peer keeps looping deadlocks both. Record the first
                        # failure, then run the remaining schedule barrier-only.
                        if err is not None:
                            continue
                        try:
                            t0 = time.perf_counter()
                            out = self._ops.rdna_ar_all_reduce(self.handle, inp)
                            torch.accelerator.synchronize(dev_idx)
                            dt = time.perf_counter() - t0
                            if debug:
                                print(
                                    f"[rdna_ar rank{self.rank}] trial{trial} rep{rep} "
                                    f"call {dt * 1e3:.1f}ms",
                                    flush=True,
                                )
                            if self._ops.rdna_ar_timed_out(self.handle):
                                err = (
                                    f"spin-cap timeout in self-test trial {trial} "
                                    f"rep {rep} ({dt * 1e3:.0f} ms)"
                                )
                            elif not bool((out == expect).all()):
                                got = out.float().mean().item()
                                err = (
                                    f"wrong result in self-test trial {trial} "
                                    f"rep {rep}: "
                                    f"mean {got:.2f}, expected {expect:.1f}"
                                )
                            elif rep > 0:
                                times.append(dt)
                        except Exception as e:  # noqa: BLE001
                            err = str(e)
                    if err is None:
                        best = min(times)
                        if best > 0.05:
                            err = (
                                f"self-test trial {trial} best {best * 1e3:.1f} ms "
                                f"of {REPEATS} for {numel * 2} bytes "
                                f"(all: {', '.join(f'{t * 1e3:.1f}' for t in times)} "
                                "ms; P2P too slow, RCCL will be faster)"
                            )
        except Exception as e:  # noqa: BLE001
            err = str(e)
        return err

    def should_use(self, inp: torch.Tensor) -> bool:
        return (not self.disabled) and self._ops.rdna_ar_can(self.handle, inp)

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        return self._ops.rdna_ar_all_reduce(self.handle, inp)

    def timed_out(self) -> bool:
        return (not self.disabled) and self._ops.rdna_ar_timed_out(self.handle)
