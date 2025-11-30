"""UNGAR Command Line Interface.

Unified entrypoint for training, analysis, and experiment management.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ungar.analysis.overlays import aggregate_overlays, load_overlays, save_aggregation
from ungar.analysis.plots import plot_learning_curve, plot_overlay_heatmap
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
        print("No runs directory found.")
        return

    runs = []
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
        except Exception:
            continue

    # Sort by timestamp descending
    runs.sort(key=lambda x: x.timestamp, reverse=True)

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
        import matplotlib  # noqa: F401
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
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is required for plotting. Install with `pip install ungar[viz]`.")
        sys.exit(1)

    run_path = args.run
    out_dir = Path(args.out_dir)
    method = args.method

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading overlays from {run_path}...")
    overlays = load_overlays(run_path)
    if not overlays:
        print("No overlays found.")
        return

    print(f"Aggregating {len(overlays)} overlays using {method}...")
    agg_map = aggregate_overlays(overlays, method=method)

    json_path = out_dir / "overlay_aggregated.json"
    save_aggregation(agg_map, json_path, label=f"aggregated_{method}")
    print(f"Saved aggregated JSON to {json_path}")

    png_path = out_dir / "overlay_heatmap.png"
    plot_overlay_heatmap(agg_map, out_path=png_path, title=f"Overlay Heatmap ({method})")
    print(f"Saved heatmap PNG to {png_path}")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="UNGAR: Universal Neural Grid for Analysis and Research"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List Runs
    subparsers.add_parser("list-runs", help="List training runs")

    # Show Run
    show_parser = subparsers.add_parser("show-run", help="Show run details")
    show_parser.add_argument("run_id", help="Run ID or partial directory name")

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
        "--method", default="mean", choices=["mean", "max", "std"], help="Aggregation method"
    )

    args = parser.parse_args()

    if args.command == "list-runs":
        cmd_list_runs(args)
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
