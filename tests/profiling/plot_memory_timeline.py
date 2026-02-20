"""
Script to plot memory timeline evolution from profiling traces.
Reads memory traces from json files and plots them on a single graph.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend to avoid tkinter dependency
import matplotlib.pyplot as plt
import numpy as np
from paths import TRACES_DIR


@dataclass
class MemoryFrame:
    timestamp: int
    total_allocated: int  # in bytes
    device_type: int  # 0 for CPU, 1 for CUDA
    device_id: int  # -1 for CPU, 0+ for CUDA devices

    @staticmethod
    def from_event(event: dict) -> "MemoryFrame":
        args = event["args"]
        return MemoryFrame(
            timestamp=event["ts"],
            total_allocated=args.get("Total Allocated"),
            device_type=args.get("Device Type"),
            device_id=args.get("Device Id"),
        )


def extract_memory_timelines(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path) as f:
        data = json.load(f)

    events = data["traceEvents"]
    print(f"Total events in trace: {len(events):,}")
    print("Extracting memory frames...")

    frames = [MemoryFrame.from_event(e) for e in events if e["name"] == "[memory]"]

    # Separate CPU (device_type=0) and CUDA (device_type=1) frames
    cpu_frames = [f for f in frames if f.device_type == 0]
    cuda_frames = [f for f in frames if f.device_type == 1]

    cpu_frames.sort(key=lambda frame: frame.timestamp)
    cuda_frames.sort(key=lambda frame: frame.timestamp)

    print(f"Found {len(cpu_frames)} CPU memory frames and {len(cuda_frames)} CUDA memory frames")

    cpu_timeline = np.array([[f.timestamp, f.total_allocated] for f in cpu_frames])
    cuda_timeline = np.array([[f.timestamp, f.total_allocated] for f in cuda_frames])

    return cpu_timeline, cuda_timeline


def plot_memory_timelines(experiment: str, folders: list[str]) -> None:
    cpu_timelines = []
    cuda_timelines = []
    for folder in folders:
        path = TRACES_DIR / folder / f"{experiment}.json"
        cpu_timeline, cuda_timeline = extract_memory_timelines(path)
        cpu_timelines.append(cpu_timeline)
        cuda_timelines.append(cuda_timeline)

    fig, (ax_cuda, ax_cpu) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    start_times = [
        min(cpu_tl[0, 0], cuda_tl[0, 0]) if len(cuda_tl) > 0 else cpu_tl[0, 0]
        for cpu_tl, cuda_tl in zip(cpu_timelines, cuda_timelines, strict=True)
    ]

    # Plot CUDA memory (top subplot)
    for folder, cuda_timeline, start_time in zip(folders, cuda_timelines, start_times, strict=True):
        if len(cuda_timeline) > 0:
            time = (cuda_timeline[:, 0] - start_time) // 1000  # Convert to ms starting at 0
            memory = cuda_timeline[:, 1]
            ax_cuda.plot(time, memory, label=folder, linewidth=1.5)

    ax_cuda.set_xlabel("Time (ms)", fontsize=12)
    ax_cuda.set_ylabel("CUDA Memory (bytes)", fontsize=12)
    ax_cuda.set_title(f"CUDA Memory Timeline: {experiment}", fontsize=14, fontweight="bold")
    ax_cuda.legend(loc="best", fontsize=11)
    ax_cuda.grid(True, alpha=0.3)
    ax_cuda.set_ylim(bottom=0)

    # Plot CPU memory (bottom subplot)
    for folder, cpu_timeline, start_time in zip(folders, cpu_timelines, start_times, strict=True):
        time = (cpu_timeline[:, 0] - start_time) // 1000  # Convert to ms starting at 0
        memory = cpu_timeline[:, 1]
        ax_cpu.plot(time, memory, label=folder, linewidth=1.5)

    ax_cpu.set_xlabel("Time (ms)", fontsize=12)
    ax_cpu.set_ylabel("CPU Memory (bytes)", fontsize=12)
    ax_cpu.set_title(f"CPU Memory Timeline: {experiment}", fontsize=14, fontweight="bold")
    ax_cpu.legend(loc="best", fontsize=11)
    ax_cpu.grid(True, alpha=0.3)
    ax_cpu.set_ylim(bottom=0)

    fig.tight_layout()

    output_dir = Path(TRACES_DIR / "memory_timelines")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{experiment}.png"
    print(f"\nSaving plot to: {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print("Plot saved successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot memory timeline from profiling traces.")
    parser.add_argument(
        "experiment",
        type=str,
        help="Name of the experiment under profiling (e.g., 'WithTransformerLarge()-bs4-cpu')",
    )
    parser.add_argument(
        "folders",
        nargs="+",
        type=str,
        help="Folder names containing the traces (e.g., autojac_old autojac_new)",
    )

    args = parser.parse_args()

    return plot_memory_timelines(args.experiment, args.folders)


if __name__ == "__main__":
    main()
