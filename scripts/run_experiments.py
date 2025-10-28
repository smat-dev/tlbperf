#!/usr/bin/env python3
import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def resolve_timeout_prefix() -> list[str]:
    env_override = os.environ.get("TLB_TIMEOUT_PREFIX")
    if env_override:
        return shlex.split(env_override)
    timeout_binary = shutil.which(os.environ.get("TLB_TIMEOUT_BINARY", "timeout"))
    if not timeout_binary:
        return []
    duration = os.environ.get("TLB_TIMEOUT_SECONDS", "300s")
    return [timeout_binary, duration, "--"]


TIMEOUT_PREFIX = resolve_timeout_prefix()
MAX_HUGETLB_2M_BYTES = 32 * 1024**3  # 32 GiB
MAX_HUGETLB_1G_BYTES = 16 * 1024**3  # 16 GiB


class CpuFreqPin:
    def __init__(self) -> None:
        self.entries: list[dict[str, str | Path]] = []

    def __enter__(self) -> "CpuFreqPin":
        self.entries = self._capture()
        if self.entries:
            print("[freq] Pinning CPU frequency (sudo required)")
            self._apply()
        else:
            print("[freq] CPU frequency scaling interface not found; skipping pin")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.entries:
            print("[freq] Restoring original CPU frequency configuration")
            self._restore()

    @staticmethod
    def _write(path: Path, value: str) -> None:
        subprocess.run(
            ["sudo", "tee", str(path)],
            input=f"{value}\n".encode(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _capture(self) -> list[dict[str, str | Path]]:
        cpu_root = Path("/sys/devices/system/cpu")
        entries: list[dict[str, str | Path]] = []
        for cpu_dir in sorted(cpu_root.glob("cpu[0-9]*")):
            cpufreq_dir = cpu_dir / "cpufreq"
            if not cpufreq_dir.is_dir():
                continue
            try:
                governor = (cpufreq_dir / "scaling_governor").read_text().strip()
                min_freq = (cpufreq_dir / "scaling_min_freq").read_text().strip()
                max_freq = (cpufreq_dir / "scaling_max_freq").read_text().strip()
            except FileNotFoundError:
                continue
            epp_path = cpufreq_dir / "energy_performance_preference"
            epp_value = None
            if epp_path.exists():
                try:
                    epp_value = epp_path.read_text().strip()
                except FileNotFoundError:
                    epp_value = None
            entries.append(
                {
                    "dir": cpufreq_dir,
                    "governor": governor,
                    "min": min_freq,
                    "max": max_freq,
                    "epp": epp_value,
                }
            )
        return entries

    def _apply(self) -> None:
        for entry in self.entries:
            dir_path: Path = entry["dir"]  # type: ignore[assignment]
            if entry["epp"] is not None:
                self._write(dir_path / "energy_performance_preference", "performance")
            self._write(dir_path / "scaling_governor", "performance")
        for entry in self.entries:
            dir_path: Path = entry["dir"]  # type: ignore[assignment]
            self._write(dir_path / "scaling_min_freq", entry["max"])  # type: ignore[arg-type]

    def _restore(self) -> None:
        for entry in self.entries:
            dir_path: Path = entry["dir"]  # type: ignore[assignment]
            self._write(dir_path / "scaling_min_freq", entry["min"])  # type: ignore[arg-type]
            self._write(dir_path / "scaling_governor", entry["governor"])  # type: ignore[arg-type]
            if entry["epp"] is not None:
                self._write(dir_path / "energy_performance_preference", entry["epp"])  # type: ignore[arg-type]


class HugePageManager:
    def __init__(self) -> None:
        self.current_2m: int | None = None
        self.current_1g: int | None = None
        self.path_1g = Path("/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages")

    def set_2m(self, count: int) -> None:
        if self.current_2m == count:
            return
        subprocess.run(
            ["sudo", "sysctl", "-w", f"vm.nr_hugepages={count}"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        self.current_2m = count

    def set_1g(self, count: int) -> None:
        if self.current_1g == count:
            return
        subprocess.run(
            ["sudo", "tee", str(self.path_1g)],
            input=f"{count}\n".encode(),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.current_1g = count

    def release_all(self) -> None:
        self.set_1g(0)
        self.set_2m(0)


def event_supported(event: str) -> bool:
    result = subprocess.run(
        ["perf", "stat", "-e", event, "--", "true"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def resolve_event_map(event_groups: Dict[str, List[str]]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for key, candidates in event_groups.items():
        for event in candidates:
            if event_supported(event):
                resolved[key] = event
                break
        if key not in resolved:
            print(f"[perf] no supported events found for {key}")
    return resolved


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Run TLB blog experiments.")
    parser.add_argument("--steps", type=int, default=200_000, help="Pointer chase steps per experiment")
    parser.add_argument("--seed", type=int, default=1337, help="PRNG seed for reproducibility")
    parser.add_argument(
        "--block-vertices",
        type=int,
        default=1 << 22,
        help="Vertices per block for blocked mode (default ≈256 MiB tiles)",
    )
    parser.add_argument("--block-steps", type=int, default=200_000, help="Steps before switching blocks")
    parser.add_argument(
        "--sizes",
        type=str,
        default="4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728,268435456,536870912",
        help="Comma separated vertex counts to test (defaults to power-of-two working sets up to 32 GiB)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run experiments even if output exists")
    parser.add_argument("--exec", type=Path, default=SCRIPT_DIR / "tlb_walk", help="Path to tlb_walk binary")
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "data", help="Directory for outputs")
    args = parser.parse_args(argv)

    tlb_exec = args.exec
    if not tlb_exec.exists():
        raise SystemExit(f"Cannot find tlb_walk executable at {tlb_exec}")

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    perf_dir = outdir / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)

    event_groups = {
        "walk_pending": ["dtlb_load_misses.walk_pending"],
        "walks": ["dtlb_load_misses.miss_causes_a_walk"],
        "loads": ["mem_inst_retired.all_loads"],
        "llc_miss": ["mem_load_retired.l3_miss", "mem_inst_retired.l3_miss", "longest_lat_cache.miss"],
        "dtlb_load_miss": ["dTLB-load-misses", "cpu/dTLB-load-misses/"],
    }
    selected_events = resolve_event_map(event_groups)
    perf_events = list(selected_events.values())
    if perf_events:
        printable = ", ".join(f"{k}={v}" for k, v in selected_events.items())
        print(f"[perf] collecting events: {printable}")
    else:
        print("[perf] no supported perf events found; skipping perf stat collection")

    uid = os.getuid()
    gid = os.getgid()

    bytes_per_vertex = 64
    huge = HugePageManager()
    huge.set_1g(0)
    huge.set_2m(0)

    def run(mode: str, size: int, extra_flags: list[str], suffix: str) -> None:
        name = f"{mode}_v{size}{suffix}.csv"
        out_path = outdir / name
        perf_path = perf_dir / name.replace(".csv", "_perf.csv")
        warmup_out_path = outdir / name.replace(".csv", "_warmup.csv")
        warmup_perf_path = perf_dir / name.replace(".csv", "_warmup_perf.csv")
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path}")
            return

        payload_cmd = [
            str(tlb_exec),
            "--mode", mode,
            "--size", str(size),
            "--steps", str(args.steps),
            "--seed", str(args.seed),
            "--output", str(out_path),
        ] + extra_flags

        needs_root = any(flag in {"--hugetlb-2m", "--hugetlb-1g"} for flag in extra_flags)

        full_cmd: list[str] = payload_cmd
        if perf_events:
            perf_cmd = [
                "perf", "stat", "-x", ",",
                "--no-scale",
                "-o", str(perf_path),
            ]
            for event in perf_events:
                perf_cmd.extend(["-e", event])
            perf_cmd.append("--")
            perf_cmd.extend(payload_cmd)
            full_cmd = perf_cmd

        if needs_root:
            full_cmd = ["sudo", "-E", *full_cmd]

        env = os.environ.copy()
        env.setdefault("CGT_MEMORY_LIMIT", "max")
        timeout_prefix: list[str] = TIMEOUT_PREFIX

        exec_sequence = full_cmd
        if timeout_prefix:
            exec_sequence = [*timeout_prefix, *full_cmd]

        print(f"[run] {' '.join(exec_sequence)}")

        subprocess.run(exec_sequence, check=True, env=env)
        if needs_root:
            subprocess.run(["sudo", "chown", f"{uid}:{gid}", str(out_path)], check=True)
            subprocess.run(["sudo", "chown", f"{uid}:{gid}", str(perf_path)], check=True)

        # Warm-up-only perf collection (steps=0) to measure initialization overhead.
        warmup_cmd = [
            str(tlb_exec),
            "--mode", mode,
            "--size", str(size),
            "--steps", "0",
            "--seed", str(args.seed),
            "--output", str(warmup_out_path),
        ] + extra_flags
        if perf_events:
            warmup_perf_cmd = [
                "perf", "stat", "-x", ",",
                "--no-scale",
                "-o", str(warmup_perf_path),
            ]
            for event in perf_events:
                warmup_perf_cmd.extend(["-e", event])
            warmup_perf_cmd.append("--")
            warmup_perf_cmd.extend(warmup_cmd)
            warmup_full_cmd = warmup_perf_cmd if not needs_root else ["sudo", "-E", *warmup_perf_cmd]
        else:
            warmup_full_cmd = warmup_cmd if not needs_root else ["sudo", "-E", *warmup_cmd]

        warmup_exec = warmup_full_cmd
        if timeout_prefix:
            warmup_exec = [*timeout_prefix, *warmup_full_cmd]
        print(f"[warmup] {' '.join(warmup_exec)}")
        subprocess.run(warmup_exec, check=True, env=env)
        if needs_root:
            if warmup_out_path.exists():
                subprocess.run(["sudo", "chown", f"{uid}:{gid}", str(warmup_out_path)], check=True)
            if perf_events and warmup_perf_path.exists():
                subprocess.run(["sudo", "chown", f"{uid}:{gid}", str(warmup_perf_path)], check=True)

        if warmup_out_path.exists():
            warmup_out_path.unlink()
        if perf_events and warmup_perf_path.exists():
            # keep for subtraction
            pass

    sizes_bytes = {size: size * bytes_per_vertex for size in sizes}
    working_set_bytes = list(sizes_bytes.values())
    max_bytes = max(working_set_bytes) if working_set_bytes else 0

    two_mebibytes = 2 * 1024 * 1024
    max_2m_pages_limit = MAX_HUGETLB_2M_BYTES // two_mebibytes
    target_2m_pages = math.ceil(max_bytes / two_mebibytes) if max_bytes else 0
    if target_2m_pages > max_2m_pages_limit:
        print(
            f"[warn] Required 2 MiB pages ({target_2m_pages}) exceeds limit {max_2m_pages_limit}; "
            f"capping to {max_2m_pages_limit}"
        )
        target_2m_pages = max_2m_pages_limit
    MIN_2M_PAGES = 16

    one_gib = 1 << 30
    max_1g_pages_limit = MAX_HUGETLB_1G_BYTES // one_gib

    def required_1g_pages(bytes_val: int) -> int:
        if bytes_val == 0:
            return 0
        return max(1, (bytes_val + one_gib - 1) // one_gib)

    pages_1g_per_size = {size: required_1g_pages(total) for size, total in sizes_bytes.items()}
    max_1g_pages = max(pages_1g_per_size.values()) if pages_1g_per_size else 0

    metadata = {
        "steps": args.steps,
        "seed": args.seed,
        "block_vertices_requested": args.block_vertices,
        "block_steps_requested": args.block_steps,
        "sizes": sizes,
        "perf_events": selected_events,
        "working_set_bytes": sizes_bytes,
        "hugetlb_2m_pages_limit": max_2m_pages_limit,
        "hugetlb_2m_pages_max_requested": max(target_2m_pages, MIN_2M_PAGES) if max_bytes else 0,
        "hugetlb_1g_pages_limit": max_1g_pages_limit,
        "hugetlb_1g_pages_required": {
            str(size): pages for size, pages in sorted(pages_1g_per_size.items())
        },
        "hugetlb_1g_pages_max": max_1g_pages,
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    with CpuFreqPin():
        # Standard 4 KiB pages
        for size in sizes:
            run("baseline", size, [], "")

        # Explicit 2 MiB pages
        for size in sizes:
            required_pages = math.ceil(sizes_bytes[size] / two_mebibytes)
            required_pages = max(required_pages, MIN_2M_PAGES)
            required_pages = min(required_pages, max_2m_pages_limit)
            huge.set_2m(required_pages)
            run("baseline", size, ["--hugetlb-2m"], "_hugetlb2m")
        huge.set_2m(0)

        # Explicit 1 GiB pages
        if max_1g_pages:
            for size in sizes:
                pages_needed = pages_1g_per_size.get(size, 0)
                if pages_needed == 0:
                    continue
                if pages_needed > max_1g_pages_limit:
                    print(
                        f"[skip] {size=} requires {pages_needed}x1GiB pages; "
                        f"limit is {max_1g_pages_limit}, skipping"
                    )
                    continue
                huge.set_1g(pages_needed)
                run("baseline", size, ["--hugetlb-1g"], "_hugetlb1g")
            huge.set_1g(0)

        # Blocked traversal (4 KiB pages)
        def nearest_pow2_leq(x: int) -> int:
            if x <= 0:
                return 1
            return 1 << (x.bit_length() - 1)

        for size in sizes:
            block_vertices = nearest_pow2_leq(min(args.block_vertices, size))
            block_steps = min(args.block_steps, args.steps)
            run(
                "blocked",
                size,
                ["--block-vertices", str(block_vertices), "--block-steps", str(block_steps)],
                f"_b{block_vertices}",
            )

    huge.release_all()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
