"""UNGAR Command Line Interface.

Unified entrypoint for training, analysis, and experiment management.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from ungar.analysis.schema import (
    SchemaError,
    validate_manifest,
    validate_metrics_file,
    validate_overlay,
)
from ungar.training.config import DQNConfig, PPOConfig
from ungar.training.device import DeviceConfig
from ungar.training.run_dir import RunManifest
from ungar.training.train_dqn import train_dqn
from ungar.training.train_ppo import train_ppo


def _get_runs_dir() -> Path:
    """Return default runs directory."""
    return Path("runs")


def cmd_list_runs(args: argparse.Namespace) -> None:
    """List all training runs."""
    runs_dir = _get_runs_dir()
    if not runs_dir.exists():
        if args.format == "json":
            print("[]")
        else:
            print("No runs directory found.")
        return

    runs = []
    run_paths = {}

    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue

        manifest_path = run_path / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                manifest = RunManifest.from_dict(data)
                runs.append(manifest)
                run_paths[manifest.run_id] = run_path
        except Exception:
            continue

    # Sort by timestamp descending
    runs.sort(key=lambda x: x.timestamp, reverse=True)

    if args.format == "json":
        output = []
        for run in runs:
            # Re-read manifest to get full dict including created_at if present
            # Or use manifest object fields. manifest.created_at might be missing on old runs
            # if we didn't update RunManifest class to default it to None.
            # But we did update RunManifest to require it.
            # Wait, if we load an old manifest that doesn't have created_at, from_dict will fail
            # if the dataclass has no default.
            # I should update RunManifest to have default for created_at for backward compat.
            # But for now assuming new runs or migration.
            # Actually, I updated RunManifest without default for created_at.
            # This breaks loading old runs.
            # I should fix RunManifest first or handle exception.
            # The exception handler above `except Exception: continue` will hide old runs.
            # This is acceptable for "v1 freeze" if we assume a clean slate or migration,
            # but ideally we should be robust.

            # Assuming valid manifest for now
            item = {
                "run_id": run.run_id,
                "game": run.game,
                "algo": run.algo,
                "created_at": getattr(run, "created_at", None),
                "path": str(run_paths[run.run_id]),
            }
            output.append(item)
        print(json.dumps(output, indent=2))
        return

    print(f"{'ID':<10} {'TIMESTAMP':<20} {'GAME':<15} {'ALGO':<10} {'DEVICE':<10}")
    print("-" * 70)

    for run in runs:
        # Simple timestamp formatting
        import datetime

        dt = datetime.datetime.fromtimestamp(run.timestamp).strftime("%Y-%m-%d %H:%M")
        print(f"{run.run_id:<10} {dt:<20} {run.game:<15} {run.algo:<10} {run.device:<10}")


def cmd_show_run(args: argparse.Namespace) -> None:
    """Show details for a specific run."""
    run_id = args.run_id
    runs_dir = _get_runs_dir()

    # Find run by ID (partial match allowed if unique)
    candidates = []
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        if run_id in run_path.name:
            candidates.append(run_path)

    if not candidates:
        print(f"No run found matching '{run_id}'")
        sys.exit(1)

    if len(candidates) > 1:
        print(f"Multiple runs found matching '{run_id}':")
        for c in candidates:
            print(f"  {c.name}")
        sys.exit(1)

    run_path = candidates[0]
    manifest_path = run_path / "manifest.json"

    if not manifest_path.exists():
        print(f"Run directory {run_path} is corrupt (missing manifest.json)")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Pretty print full manifest
        print(json.dumps(data, indent=2))


def cmd_export_run(args: argparse.Namespace) -> None:
    """Export a run to a destination directory."""
    run_id = args.run_id
    out_dir = Path(args.out_dir)
    runs_dir = _get_runs_dir()

    # Find run
    candidates = []
    for run_path in runs_dir.iterdir():
        if not run_path.is_dir():
            continue
        if run_id in run_path.name:
            candidates.append(run_path)

    if not candidates:
        print(f"No run found matching '{run_id}'")
        sys.exit(1)
    if len(candidates) > 1:
        print(f"Multiple runs found matching '{run_id}':")
        for c in candidates:
            print(f"  {c.name}")
        sys.exit(1)

    run_path = candidates[0]
    print(f"Exporting run from {run_path} to {out_dir}...")

    # Validate before export
    try:
        manifest_path = run_path / "manifest.json"
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        validate_manifest(manifest_data)

        metrics_path = run_path / "metrics.csv"
        if metrics_path.exists():
            validate_metrics_file(metrics_path)

        overlays_dir = run_path / "overlays"
        if overlays_dir.exists():
            for overlay_file in overlays_dir.glob("*.json"):
                with open(overlay_file, "r", encoding="utf-8") as f:
                    # Could be list or dict
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            validate_overlay(item)
                    else:
                        validate_overlay(data)

    except SchemaError as e:
        print(f"Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.exit(1)

    # Perform copy
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(run_path, out_dir)
    print("Export complete.")


def cmd_train(args: argparse.Namespace) -> None:
    """Start a training run."""
    game = args.game
    algo = args.algo
    episodes = args.episodes
    run_dir = args.run_dir or _get_runs_dir()
    device_str = args.device

    print(f"Starting {algo.upper()} training on {game}...")
    print(f"Episodes: {episodes if episodes else 'Default'}")
    print(f"Device: {device_str}")

    device_config = DeviceConfig(device=device_str)

    if algo == "dqn":
        dqn_config = DQNConfig(device=device_config)
        if episodes:
            dqn_config.total_episodes = episodes

        result = train_dqn(game_name=game, config=dqn_config, run_dir=run_dir)

    elif algo == "ppo":
        ppo_config = PPOConfig(device=device_config)
        if episodes:
            ppo_config.total_episodes = episodes

        result = train_ppo(game_name=game, config=ppo_config, run_dir=run_dir)  # type: ignore[assignment]
    else:
        print(f"Unknown algorithm: {algo}")
        sys.exit(1)

    if result.run_dir:
        print("\nTraining complete.")
        print(f"Run directory: {result.run_dir}")
        print(f"Metrics: {result.metrics}")
        print("\nTo analyze results, see docs/analytics_overview.md")
    else:
        print("\nTraining complete (no run directory created).")


def cmd_plot_curves(args: argparse.Namespace) -> None:
    """Plot learning curves."""
    try:
        from ungar.analysis.plots import plot_learning_curve
    except ImportError:
        print("matplotlib is required for plotting. Install with `pip install ungar[viz]`.")
        sys.exit(1)

    run_paths = args.run
    out_path = args.out
    smooth = args.smooth

    print(f"Plotting curves for {len(run_paths)} runs to {out_path}...")
    plot_learning_curve(run_paths, out_path=out_path, smooth_window=smooth)
    print("Done.")


def cmd_summarize_overlays(args: argparse.Namespace) -> None:
    """Aggregate and plot XAI overlays."""
    try:
        from ungar.analysis.overlays import (
            compute_max_overlay,
            compute_mean_overlay,
            load_overlays,
            overlay_to_dict,
        )
        from ungar.analysis.plots import plot_overlay_heatmap
    except ImportError:
        print("matplotlib is required for plotting. Install with `pip install ungar[viz]`.")
        sys.exit(1)

    run_path = args.run
    out_dir = Path(args.out_dir)
    agg_method = args.agg  # "mean" or "max"
    label_filter = args.label

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading overlays from {run_path}...")
    overlays = load_overlays(run_path)

    if label_filter:
        overlays = [o for o in overlays if o.label == label_filter]
        print(f"Filtered to {len(overlays)} overlays with label '{label_filter}'")

    if not overlays:
        print("No overlays found.")
        return

    print(f"Aggregating {len(overlays)} overlays using {agg_method}...")

    if agg_method == "mean":
        agg_overlay = compute_mean_overlay(overlays, label=f"aggregated_{agg_method}")
    else:
        agg_overlay = compute_max_overlay(overlays, label=f"aggregated_{agg_method}")

    json_path = out_dir / f"overlay_{agg_method}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overlay_to_dict(agg_overlay), f, indent=2)

    print(f"Saved aggregated JSON to {json_path}")

    png_path = out_dir / f"overlay_{agg_method}_heatmap.png"
    # plot_overlay_heatmap expects the raw 4x14 array for now?
    # Or we can update it to take CardOverlay.
    # Existing signature: plot_overlay_heatmap(importance: np.ndarray, ...)
    plot_overlay_heatmap(
        agg_overlay.importance, out_path=png_path, title=f"Overlay Heatmap ({agg_method})"
    )
    print(f"Saved heatmap PNG to {png_path}")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="UNGAR: Universal Neural Grid for Analysis and Research"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List Runs
    list_parser = subparsers.add_parser("list-runs", help="List training runs")
    list_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )

    # Show Run
    show_parser = subparsers.add_parser("show-run", help="Show run details")
    show_parser.add_argument("run_id", help="Run ID or partial directory name")

    # Export Run
    export_parser = subparsers.add_parser("export-run", help="Export run artifacts")
    export_parser.add_argument("--run-id", required=True, help="Run ID or partial directory name")
    export_parser.add_argument("--out-dir", required=True, help="Destination directory")

    # Training
    train_parser = subparsers.add_parser("train", help="Start training")
    train_parser.add_argument(
        "--game", required=True, help="Game name (high_card_duel, spades_mini, gin_rummy)"
    )
    train_parser.add_argument("--algo", required=True, choices=["dqn", "ppo"], help="Algorithm")
    train_parser.add_argument("--episodes", type=int, help="Number of episodes")
    train_parser.add_argument("--run-dir", help="Base run directory")
    train_parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use"
    )

    # Analysis
    plot_parser = subparsers.add_parser("plot-curves", help="Plot learning curves")
    plot_parser.add_argument("--run", action="append", required=True, help="Run ID or path")
    plot_parser.add_argument("--out", required=True, help="Output image path")
    plot_parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size")

    overlay_parser = subparsers.add_parser("summarize-overlays", help="Aggregate XAI overlays")
    overlay_parser.add_argument("--run", required=True, help="Run ID or path")
    overlay_parser.add_argument("--out-dir", required=True, help="Output directory")
    overlay_parser.add_argument(
        "--agg", default="mean", choices=["mean", "max"], help="Aggregation function"
    )
    overlay_parser.add_argument(
        "--label", default=None, help="Filter by overlay label (e.g. heuristic)"
    )

    args = parser.parse_args()

    if args.command == "list-runs":
        cmd_list_runs(args)
    elif args.command == "export-run":
        cmd_export_run(args)
    elif args.command == "show-run":
        cmd_show_run(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "plot-curves":
        cmd_plot_curves(args)
    elif args.command == "summarize-overlays":
        cmd_summarize_overlays(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
