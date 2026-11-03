"""Profile a Triton workload with Nsight Compute on Modal.

    MODAL_GPU=A100-80GB modal run triton/profile_runner.py --script 1_fused.py
"""

import csv
import importlib.util
import io
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path


LOCAL_DIR = Path(__file__).resolve().parent
REMOTE_DIR = Path("/root/kernels")
CHILD_ENV = "TRITON_NCU_CHILD"
REPORT_ENV = "TRITON_PROFILE_REPORT_DIR"
SPILL_METRICS = (
    "sass__inst_executed_register_spilling",
    "sass__inst_executed_local_loads",
    "sass__inst_executed_local_stores",
)
SUMMARY_METRICS = {
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__registers_per_thread_allocated",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__shared_mem_per_block",
    "launch__shared_mem_per_block_allocated",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
}


def resolve_script(root, script):
    root = root.resolve()
    path = (root / script).resolve()
    if not path.is_relative_to(root) or path.suffix != ".py":
        raise ValueError("Choose a Python script inside the triton directory.")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name in ("profile_runner.py", "modal_runner.py"):
        raise ValueError("Choose a kernel script as the profiling target.")
    return path


def find_ncu(install_root=Path("/opt/nvidia/nsight-compute")):
    """Find a native, runnable profiler rather than an arbitrary target binary."""
    machine = platform.machine().lower()
    elf_machine = {"x86_64": 62, "amd64": 62, "aarch64": 183, "arm64": 183}.get(machine)
    on_path = shutil.which("ncu")
    # Prefer the installation's launcher over the bundled target executables.
    candidates = ([Path(on_path)] if on_path else [])
    candidates += sorted(install_root.glob("*/ncu"), reverse=True)
    candidates += sorted(install_root.glob("*/target/*/ncu"), reverse=True)
    seen, failures = set(), []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            with candidate.open("rb") as binary:
                header = binary.read(20)
            if header[:4] == b"\x7fELF" and len(header) >= 20 and elf_machine is not None:
                byteorder = "little" if header[5] == 1 else "big"
                target_machine = int.from_bytes(header[18:20], byteorder)
                if target_machine != elf_machine:
                    failures.append(f"{candidate}: ELF machine {target_machine} does not match {machine}")
                    continue
            # This also validates shell launchers, shared libraries, and permissions.
            version = subprocess.check_output(
                [str(candidate), "--version"], text=True, stderr=subprocess.STDOUT, timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            failures.append(f"{candidate}: {exc}")
            continue
        return str(candidate), version
    details = "\n".join(failures) or f"No ncu executable found on PATH or under {install_root}."
    raise RuntimeError(f"No runnable Nsight Compute executable for {machine}:\n{details}")


def dump_kernel_resources(compiled_kernels, output_dir):
    resources = {}
    for name, kernel in compiled_kernels.items():
        # On NVIDIA, Triton's n_spills is LOCAL_SIZE_BYTES / 4, not a dynamic
        # spill-instruction count. Confirm spilling with Nsight's SASS counters.
        n_spills = getattr(kernel, "n_spills", None)
        resources[name] = {
            "registers_per_thread": kernel.n_regs,
            "triton_n_spills": n_spills,
            "local_bytes_per_thread": None if n_spills is None else 4 * n_spills,
            "shared_bytes_per_block": kernel.metadata.shared,
            "warps_per_block": kernel.metadata.num_warps,
            "metadata": kernel.metadata._asdict(),
        }
        print(
            f"{name}: registers/thread={kernel.n_regs}, "
            f"local bytes/thread={resources[name]['local_bytes_per_thread']} "
            f"(Triton n_spills={n_spills}), shared bytes/block={kernel.metadata.shared}",
            flush=True,
        )
        for stage in ("ttir", "ttgir", "llir", "ptx", "cubin"):
            if stage in kernel.asm:
                artifact = kernel.asm[stage]
                path = output_dir / f"{name}.{stage}"
                if isinstance(artifact, bytes):
                    path.write_bytes(artifact)
                else:
                    path.write_text(str(artifact))
    return resources


def profile_child(script, args):
    import torch
    import torch.cuda.profiler
    import triton

    target = resolve_script(LOCAL_DIR, script)
    output_dir = Path(os.environ[REPORT_ENV])
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *args]
    spec = importlib.util.spec_from_file_location("profile_target", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    factory = getattr(module, "profile_workload", None)
    if not callable(factory):
        raise RuntimeError(
            f"{script} must define profile_workload(); see triton/PROFILING.md for the hook."
        )

    with torch.inference_mode():
        workload = factory()
        if callable(workload):
            workload = {"run": workload}
        if not isinstance(workload, dict) or not callable(workload.get("run")):
            raise TypeError("profile_workload() must return a callable or a dict containing callable 'run'.")
        run = workload["run"]
        compiled = workload.get("kernels", {})
        check = workload.get("check")
        if not isinstance(compiled, dict) or (check is not None and not callable(check)):
            raise TypeError("Optional 'kernels' must be a dict and 'check' must be callable.")

        # Compile and allocate outside the profiled region.
        run()
        torch.cuda.synchronize()
        resources = {
            "script": script,
            "args": args,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "workload": workload.get("metadata", {}),
            "kernels": dump_kernel_resources(compiled, output_dir),
        }
        resource_path = output_dir / "resources.json"
        resource_path.write_text(json.dumps(resources, indent=2, default=str))
        print(f"GPU: {resources['gpu']}; Torch: {torch.__version__}; Triton: {triton.__version__}", flush=True)
        if check is not None:
            resources["correctness"] = check()
            resource_path.write_text(json.dumps(resources, indent=2, default=str))
            print("Correctness check: PASS", flush=True)
        for _ in range(int(os.environ["TRITON_PROFILE_WARMUP"])):
            run()
        torch.cuda.synchronize()

        # --profile-from-start off excludes setup, correctness, and warmup.
        torch.cuda.profiler.start()
        try:
            run()
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()


def summarize_metrics(csv_path, output_dir):
    lines = [
        line for line in csv_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("==")
    ]
    reader = csv.DictReader(lines)
    rows = list(reader)
    if "Metric Name" not in (reader.fieldnames or []):
        # Raw-page CSV can have one metric per column, a units row, then one
        # data row per kernel. Normalize it to the same layout as details CSV.
        metric_columns = [name for name in (reader.fieldnames or []) if "__" in name]
        units = next((row for row in rows if not row.get("ID") and not row.get("Kernel Name")), {})
        rows = [
            {
                "ID": row.get("ID", ""),
                "Kernel Name": row["Kernel Name"],
                "Metric Name": metric,
                "Metric Unit": units.get(metric, ""),
                "Metric Value": row.get(metric, ""),
            }
            for row in rows if row.get("Kernel Name")
            for metric in metric_columns
        ]
    if not rows or not any(row.get("Kernel Name") for row in rows):
        raise RuntimeError("No profiled kernels in the report; check the filters and ncu.log.")

    fields = ("ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value")
    selected = []
    for row in rows:
        metric = row.get("Metric Name", "")
        if metric in SUMMARY_METRICS or "spill" in metric or "local" in metric:
            selected.append({key: row.get(key, "") for key in fields})
    if not selected:
        raise RuntimeError(
            "No raw metric identifiers for registers/spills/throughput found in metrics.csv. "
            "Export using --page raw --csv, or --page details --print-metric-name name."
        )
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    summary = "\n".join(
        f"{row['Kernel Name']}: {row['Metric Name']} = {row['Metric Value']} {row['Metric Unit']}"
        for row in selected
    )
    (output_dir / "summary.txt").write_text(summary + "\n")
    print("\nRegisters, spilling, local memory, occupancy, and throughput:", flush=True)
    print(summary, flush=True)


def export_report(ncu, report, output_dir):
    for filename, options in (
        ("metrics.csv", ["--page", "raw", "--csv", "--print-units", "base"]),
        ("sass.txt", ["--page", "source", "--print-source", "sass", "--metrics", ",".join(SPILL_METRICS)]),
    ):
        path = output_dir / filename
        try:
            with path.open("w") as stream:
                # Raw CSV already uses metric identifiers. --print-metric-name
                # is supported only with --page details, not raw or source.
                subprocess.run(
                    [ncu, "--import", str(report), *options],
                    stdout=stream, stderr=subprocess.STDOUT, check=True,
                )
        except subprocess.CalledProcessError as exc:
            details = path.read_text(errors="replace").strip()
            raise RuntimeError(f"Nsight export of {filename} failed: {details[-2000:]}") from exc


def archive_reports(output_dir):
    archive = io.BytesIO()
    # Modal-uploaded source files can have Unix-epoch mtimes. Clamp timestamps
    # to ZIP's supported range instead of losing the completed profiling run.
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, strict_timestamps=False,
    ) as bundle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(output_dir))
    return archive.getvalue()


def run_logged(command, log_path, env):
    with log_path.open("w") as log:
        with subprocess.Popen(
            command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            returncode = process.wait()
    if returncode:
        raise RuntimeError(f"Nsight Compute exited with status {returncode}; see {log_path.name}.")


if os.getenv(CHILD_ENV) != "1":
    import modal

    app = modal.App("triton-ncu-profiler")
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cu130")
        .pip_install("numpy==2.2.6", "matplotlib==3.10.3", "pandas==2.2.3", "pytest")
        .apt_install("wget")
        .run_commands(
            "wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb",
            "dpkg -i /tmp/cuda-keyring.deb && apt-get update && apt-get install -y nsight-compute-2026.3.1",
        )
        .add_local_dir(
            LOCAL_DIR, remote_path=str(REMOTE_DIR),
            ignore=[".venv", "__pycache__", ".git", "*.pyc", "profiles"],
        )
    )

    @app.function(image=image, gpu=os.getenv("MODAL_GPU", "A100-80GB"), timeout=1800)
    def profile(script, args, kernel_name, launch_skip, launch_count, section_set, warmup):
        output_dir = Path(tempfile.mkdtemp(prefix="triton-profile-"))
        error = None
        try:
            target = resolve_script(REMOTE_DIR, script)
            ncu, version = find_ncu()
            (output_dir / "ncu-version.txt").write_text(version)
            print(f"Nsight executable: {ncu}; container architecture: {platform.machine()}", flush=True)
            print(version, flush=True)
            # Keep sibling modules too: staged kernels import their common
            # epilogue/copy helpers, which are part of the profiled implementation.
            for source in target.parent.glob("*.py"):
                source_copy = output_dir / "source" / source.relative_to(REMOTE_DIR)
                source_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, source_copy)
            shutil.copy2(REMOTE_DIR / "profile_runner.py", output_dir / "source" / "profile_runner.py")
            report = output_dir / "profile.ncu-rep"
            command = [
                ncu, "--clock-control", "none", "--cache-control", "all",
                "--replay-mode", "kernel", "--profile-from-start", "off",
                "--kernel-name", kernel_name, "--set", section_set,
                "--metrics", ",".join(SPILL_METRICS),
                "--import-source", "yes", "--source-folders", str(REMOTE_DIR),
                "--export", str(report), "--force-overwrite",
                "--page", "details", "--print-details", "all",
                "--print-metric-name", "label-name",
            ]
            if launch_skip:
                command.extend(["--launch-skip", str(launch_skip)])
            if launch_count:
                command.extend(["--launch-count", str(launch_count)])
            command.extend([sys.executable, "-u", str(REMOTE_DIR / "profile_runner.py"), script, *args])
            (output_dir / "command.json").write_text(json.dumps(command, indent=2))
            env = {
                **os.environ, CHILD_ENV: "1", REPORT_ENV: str(output_dir),
                "TRITON_PROFILE_WARMUP": str(warmup),
            }
            run_logged(command, output_dir / "ncu.log", env)
            export_report(ncu, report, output_dir)
            summarize_metrics(output_dir / "metrics.csv", output_dir)
        except Exception as exc:
            error = str(exc)
            (output_dir / "error.txt").write_text(error + "\n")
            print(f"Profiling failed: {error}", flush=True)

        # Return reports even on failure so the local caller keeps the logs.
        result = {"archive": archive_reports(output_dir), "error": error}
        shutil.rmtree(output_dir)
        return result

    @app.function(image=image, timeout=600)
    def export_saved_report(report_bytes):
        """Read an existing report on a CPU container; no GPU collection needed."""
        with tempfile.TemporaryDirectory(prefix="triton-export-") as directory:
            root = Path(directory)
            report = root / "profile.ncu-rep"
            report.write_bytes(report_bytes)
            output_dir = root / "exports"
            output_dir.mkdir()
            error = None
            try:
                ncu, version = find_ncu()
                (output_dir / "ncu-version.txt").write_text(version)
                print(f"Exporting saved report with {ncu} (CPU only)", flush=True)
                export_report(ncu, report, output_dir)
                summarize_metrics(output_dir / "metrics.csv", output_dir)
            except Exception as exc:
                error = str(exc)
                (output_dir / "export-error.txt").write_text(error + "\n")
                print(f"Report export failed: {error}", flush=True)
            return {"archive": archive_reports(output_dir), "error": error}

    @app.local_entrypoint()
    def main(
        script: str = "swiglu_block/1_fused.py",
        args: str = "",
        kernel_name: str = "regex:.*",
        launch_skip: int = 0,
        launch_count: int = 0,
        section_set: str = "full",
        warmup: int = 3,
        report: str = "",
    ):
        if report:
            report_path = Path(report).expanduser().resolve()
            if report_path.suffix != ".ncu-rep" or not report_path.is_file():
                raise ValueError("--report must point to an existing .ncu-rep file.")
            result = export_saved_report.remote(report_path.read_bytes())
            output_dir = report_path.parent
        else:
            target = resolve_script(LOCAL_DIR, script)
            if min(launch_skip, launch_count, warmup) < 0:
                raise ValueError("Launch counts and warmup must be nonnegative.")
            script = str(target.relative_to(LOCAL_DIR))
            result = profile.remote(
                script, shlex.split(args), kernel_name, launch_skip, launch_count, section_set, warmup,
            )
            output_dir = LOCAL_DIR / "profiles" / Path(script).with_suffix("") / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(result["archive"])) as bundle:
            bundle.extractall(output_dir)
        print(f"Reports saved to {output_dir}")
        if result["error"]:
            raise RuntimeError(f"{result['error']} Logs saved to {output_dir}.")
        print(f"Open {output_dir / 'profile.ncu-rep'} in Nsight Compute.")


if __name__ == "__main__" and os.getenv(CHILD_ENV) == "1":
    profile_child(sys.argv[1], sys.argv[2:])
