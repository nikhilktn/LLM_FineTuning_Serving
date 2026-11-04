"""Step 6: save a training run's TensorBoard curves to results/ and plot them.

  # on the server, in the train venv: export the curves next to the other results
  python scripts/plot_training.py --run-dir outputs/qwen7b-lora --label lora-r16
  # anywhere with matplotlib: redraw the chart from the exported JSON
  python scripts/plot_training.py --json results/train_lora-r16.json

Writes results/train_<label>.json (train_summary.json + every scalar) and
results/train_<label>.png (learning rate, loss, token accuracy, grad norm).
"""
import argparse
import glob
import json
import os

TAGS = ["train/learning_rate", "train/loss", "eval/loss", "train/mean_token_accuracy",
        "eval/mean_token_accuracy", "train/grad_norm"]


def export(run_dir, label, out_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(sorted(glob.glob(os.path.join(run_dir, "runs", "*")))[-1])
    acc.Reload()
    scalars = {}
    for tag in TAGS:
        if tag in acc.Tags()["scalars"]:
            points = {s.step: s.value for s in acc.Scalars(tag)}  # the final evaluate() repeats the last step
            scalars[tag] = sorted(points.items())
    adapter = json.load(open(os.path.join(run_dir, "final", "adapter_config.json")))
    run = {"label": label,
           "summary": json.load(open(os.path.join(run_dir, "train_summary.json"))),
           "lora": {k: adapter[k] for k in ("r", "lora_alpha", "lora_dropout", "target_modules")},
           "scalars": scalars}
    path = os.path.join(out_dir, f"train_{label}.json")
    with open(path, "w") as f:
        json.dump(run, f, indent=1)
    print(f"-> {path}")
    return run


def plot(run, out_dir, steps_per_epoch=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    train_c, eval_c, ink, muted, grid, surface = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    s = run["scalars"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=surface)
    panels = [
        (axes[0, 0], "Learning rate (×1e-4)",[("train/learning_rate", None)], False),
        (axes[0, 1], "Loss (log scale)", [("train/loss", "train"), ("eval/loss", "eval")], True),
        (axes[1, 0], "Answer-token accuracy", [("train/mean_token_accuracy", "train"),
                                               ("eval/mean_token_accuracy", "eval")], False),
        (axes[1, 1], "Gradient norm", [("train/grad_norm", None)], False),
    ]
    for ax, title, series, log in panels:
        ax.set_facecolor(surface)
        for tag, name in series:
            if tag not in s:
                continue
            x, y = zip(*s[tag])
            color = eval_c if tag.startswith("eval") else train_c
            ax.plot(x, y, color=color, linewidth=2, marker="o" if tag.startswith("eval") else None,
                    markersize=5, markeredgecolor=surface, markeredgewidth=1.5)
            if name:  # direct label at the line's end instead of a legend box
                ax.annotate(f"{name} {y[-1]:.4g}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=ink)
        if steps_per_epoch:
            ax.axvline(steps_per_epoch, color=muted, linewidth=1, linestyle=(0, (3, 3)))
            ax.annotate("epoch 2 →", (steps_per_epoch, 1), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points", fontsize=8, color=muted)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        elif tag == "train/learning_rate":
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v * 1e4:g}"))
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.margins(x=0.12)
    for ax in axes[1]:
        ax.set_xlabel("optimizer step", fontsize=9, color=muted)
    sm = run["summary"]
    fig.suptitle(f"{run['label']}: r={sm['lora_r']}, {sm['train_minutes']} min, peak {sm['peak_gpu_mem_gb']} GB, "
                 f"final eval loss {sm.get('eval_loss')}", x=0.01, ha="left", fontsize=12, color=ink)
    fig.tight_layout()
    path = os.path.join(out_dir, f"train_{run['label']}.png")
    fig.savefig(path, dpi=130, facecolor=surface)
    print(f"-> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="training output dir with runs/, final/ and train_summary.json")
    parser.add_argument("--label", help="name for this run, e.g. lora-r16")
    parser.add_argument("--json", help="replot from an exported results/train_<label>.json instead")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="draws the epoch boundary")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    if args.json:
        run = json.load(open(args.json))
    else:
        run = export(args.run_dir, args.label, args.out_dir)
    try:
        plot(run, args.out_dir, args.steps_per_epoch)
    except ImportError:
        print("matplotlib not installed; JSON exported, plot skipped")
