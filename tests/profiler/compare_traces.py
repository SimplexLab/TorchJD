"""Compare profiling traces between autojac_old and autojac_new."""

import json
from pathlib import Path


def find_event_duration(trace_data: dict, event_name: str) -> float | None:
    """Find the duration of a specific event in the trace.

    :param trace_data: The parsed JSON trace data
    :param event_name: The name of the event to find (e.g., "jac_to_grad")
    """
    events = trace_data.get("traceEvents", [])
    for event in events:
        if "name" in event and "dur" in event and event["name"].endswith(f": {event_name}"):
            return event["dur"] / 1000.0  # Convert microseconds to milliseconds
    return None


def parse_filename(filename: str) -> tuple[str, int, str]:
    """Parse model name, batch size, and device from filename.

    :param filename: The trace filename (e.g., "AlexNet()-bs4-cpu.json")
    """
    # Remove .json extension
    name = filename.replace(".json", "")
    # Split by -bs and -
    parts = name.split("-bs")
    model = parts[0].replace("()", "")
    rest = parts[1].split("-")
    batch_size = int(rest[0])
    device = rest[1]
    return model, batch_size, device


def compare_traces() -> None:
    """Compare traces between autojac_old and autojac_new directories."""
    traces_dir = Path(__file__).parent.parent.parent / "traces"
    old_dir = traces_dir / "autojac_old"
    new_dir = traces_dir / "autojac_new"

    # Collect data
    cpu_data = []
    cuda_data = []

    for old_file in sorted(old_dir.glob("*.json")):
        model, batch_size, device = parse_filename(old_file.name)
        new_file = new_dir / old_file.name

        if not new_file.exists():
            print(f"Warning: {new_file.name} not found in autojac_new")
            continue

        # Load trace files
        with open(old_file) as f:
            old_trace = json.load(f)
        with open(new_file) as f:
            new_trace = json.load(f)

        # Find durations for both events
        old_jac_to_grad = find_event_duration(old_trace, "jac_to_grad")
        new_jac_to_grad = find_event_duration(new_trace, "jac_to_grad")
        old_forward_backward = find_event_duration(old_trace, "autojac_forward_backward")
        new_forward_backward = find_event_duration(new_trace, "autojac_forward_backward")

        if old_jac_to_grad is None or new_jac_to_grad is None:
            print(f"Warning: jac_to_grad not found in {old_file.name}")
            continue
        if old_forward_backward is None or new_forward_backward is None:
            print(f"Warning: autojac_forward_backward not found in {old_file.name}")
            continue

        # Calculate differences
        diff_jac_to_grad = new_jac_to_grad - old_jac_to_grad
        diff_forward_backward = new_forward_backward - old_forward_backward
        pct_jac_to_grad = (diff_jac_to_grad / old_jac_to_grad) * 100
        pct_forward_backward = (diff_forward_backward / old_forward_backward) * 100

        row = {
            "model": model,
            "batch_size": batch_size,
            "old_jac_to_grad": old_jac_to_grad,
            "new_jac_to_grad": new_jac_to_grad,
            "diff_jac_to_grad": diff_jac_to_grad,
            "pct_jac_to_grad": pct_jac_to_grad,
            "old_forward_backward": old_forward_backward,
            "new_forward_backward": new_forward_backward,
            "diff_forward_backward": diff_forward_backward,
            "pct_forward_backward": pct_forward_backward,
        }

        if device == "cpu":
            cpu_data.append(row)
        else:
            cuda_data.append(row)

    # Print tables
    print("CPU Traces Comparison")
    print_table(cpu_data)

    print("\nCUDA Traces Comparison")
    print_table(cuda_data)


def print_table(data: list[dict]) -> None:
    """Print a formatted comparison table.

    :param data: List of row dictionaries with timing data
    """
    # Header
    header = (
        "|Model|Batch Size|Time before (jac_to_grad)|Time after (jac_to_grad)|"
        "Difference (jac_to_grad)|Time before (autojac_forward_backward)|"
        "Time after (autojac_forward_backward)|Difference (autojac_forward_backward)|"
    )
    separator = "|---|---|---|---|---|---|---|---|"

    print(header)
    print(separator)

    # Rows
    for row in data:
        # Format differences with + sign for positive values
        diff_jac = round(row["diff_jac_to_grad"])
        pct_jac = round(row["pct_jac_to_grad"])
        diff_fb = round(row["diff_forward_backward"])
        pct_fb = round(row["pct_forward_backward"])

        diff_jac_str = f"+{diff_jac}" if diff_jac > 0 else str(diff_jac)
        pct_jac_str = f"+{pct_jac}" if pct_jac > 0 else str(pct_jac)
        diff_fb_str = f"+{diff_fb}" if diff_fb > 0 else str(diff_fb)
        pct_fb_str = f"+{pct_fb}" if pct_fb > 0 else str(pct_fb)

        print(
            f"|{row['model']}|{row['batch_size']}|"
            f"{round(row['old_jac_to_grad'])} ms|"
            f"{round(row['new_jac_to_grad'])} ms|"
            f"{diff_jac_str} ms ({pct_jac_str}%)|"
            f"{round(row['old_forward_backward'])} ms|"
            f"{round(row['new_forward_backward'])} ms|"
            f"{diff_fb_str} ms ({pct_fb_str}%)|",
        )


if __name__ == "__main__":
    compare_traces()
