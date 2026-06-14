import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tokenizer results")
    parser.add_argument("--output-dir", type=str, default="results/plots")

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_scaling = subparsers.add_parser("scaling", help="Plot naive vs fast training time")
    p_scaling.add_argument("--models-dir", type=str, default="results/models")
    p_scaling.add_argument("--tokenizers", type=str, default="bpe_naive,bpe_fast,wp_naive,wp_fast")
    p_scaling.add_argument("--vocab_size", type=int, required=True)
    p_scaling.add_argument("--n_scaling",  type=str, required=True,
                           help="Comma-separated list of n_train values e.g. 10,50,100")

    p_morpho = subparsers.add_parser("morpho", help="Plot morphological metrics (BPE vs WP)")
    p_morpho.add_argument("--eval-csv", type=str, default="results/eval.csv")

    p_encode = subparsers.add_parser("encode", help="Plot compression / fertility / encode time")
    p_encode.add_argument("--eval-csv", type=str, default="results/eval.csv")

    return parser.parse_args()


def load_train_times(models_dir: str,
                     tokenizers: list[str],
                     vocab_size: int,
                     n_scaling: list[int]) -> pd.DataFrame:
    records = []
    for tokenizer in tokenizers:
        for n in n_scaling:
            meta_path = Path(models_dir) / f"{tokenizer}_v{vocab_size}_n{n}_meta.json"
            if not meta_path.exists():
                print(f"Missing: {meta_path}")
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            records.append({
                "tokenizer": tokenizer,
                "n_train":   n,
                "train_time": meta["train_time"],
            })
    return pd.DataFrame(records)


def plot_morpho(df: pd.DataFrame, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics = ["morpho_precision", "morpho_recall", "morpho_f1_morpheme", "morpho_f1_boundary"]
    metric_labels = ["Precision", "Recall", "F1 (morpheme)", "F1 (boundary)"]

    vocab_sizes = sorted(df["vocab_size"].unique())

    fig, axes = plt.subplots(1, len(vocab_sizes), figsize=(6 * len(vocab_sizes), 5), sharey=True)
    if len(vocab_sizes) == 1:
        axes = [axes]

    colors = {"bpe_fast": "tab:blue", "wp_fast": "tab:orange"}
    labels = {"bpe_fast": "BPE", "wp_fast": "WordPiece"}

    for ax, vocab in zip(axes, vocab_sizes):
        sub = df[df["vocab_size"] == vocab]

        x = range(len(metrics))
        width = 0.35

        for i, model in enumerate(sub["model"].unique()):
            row = sub[sub["model"] == model].iloc[0]
            values = [row[m] for m in metrics]
            offset = (i - 0.5) * width
            ax.bar([xi + offset for xi in x], values, width,
                   label=labels.get(model, model), color=colors.get(model, "gray"))

        ax.set_xticks(list(x))
        ax.set_xticklabels(metric_labels, rotation=15)
        ax.set_title(f"Vocab size = {vocab:,}")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()

    axes[0].set_ylabel("Score")
    fig.suptitle("Morphological alignment: BPE vs WordPiece")
    fig.tight_layout()

    output_path = Path(output_dir) / "morpho.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_encode(df: pd.DataFrame, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics = ["compression", "fertility_mean", "encode_time"]
    metric_labels = ["Compression ratio", "Fertility (mean)", "Encode time (s)"]

    vocab_sizes = sorted(df["vocab_size"].unique())

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    colors = {"bpe_fast": "tab:blue", "wp_fast": "tab:orange"}
    labels = {"bpe_fast": "BPE", "wp_fast": "WordPiece"}

    for ax, metric, metric_label in zip(axes, metrics, metric_labels):
        x = range(len(vocab_sizes))
        width = 0.35

        for i, model in enumerate(df["model"].unique()):
            sub = df[df["model"] == model].sort_values("vocab_size")
            values = [sub[sub["vocab_size"] == v][metric].iloc[0] for v in vocab_sizes]
            offset = (i - 0.5) * width
            ax.bar([xi + offset for xi in x], values, width,
                   label=labels.get(model, model), color=colors.get(model, "gray"))

        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{v:,}" for v in vocab_sizes])
        ax.set_xlabel("Vocabulary size")
        ax.set_title(metric_label)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend()

    fig.suptitle("Encoding metrics: BPE vs WordPiece")
    fig.tight_layout()

    output_path = Path(output_dir) / "encode.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def plot_scaling(df: pd.DataFrame, output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    styles = {
        "bpe_naive": ("BPE naive",  "o--", "tab:blue"),
        "bpe_fast":  ("BPE fast",  "o-",  "tab:blue"),
        "wp_naive":  ("WP naive",   "s--", "tab:orange"),
        "wp_fast":   ("WP fast",   "s-",  "tab:orange"),
    }

    for tokenizer, group in df.groupby("tokenizer"):
        label, style, color = styles.get(tokenizer, (tokenizer, "o-", "gray"))
        group = group.sort_values("n_train")
        ax.plot(group["n_train"], group["train_time"],
                style, color=color, label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Number of training articles")
    ax.set_ylabel("Training time (s)")
    ax.set_title("Temporal complexity: Fast vs Naive")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = Path(output_dir) / "scaling.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    args = parse_args()

    if args.command == "scaling":
        tokenizers = args.tokenizers.split(",")
        n_scaling = [int(n) for n in args.n_scaling.split(",")]
        df = load_train_times(args.models_dir, tokenizers, args.vocab_size, n_scaling)
        if df.empty:
            print("No data found.")
            return
        print(df.to_string(index=False))
        plot_scaling(df, args.output_dir)

    elif args.command == "morpho":
        df = pd.read_csv(args.eval_csv)
        plot_morpho(df, args.output_dir)

    elif args.command == "encode":
        df = pd.read_csv(args.eval_csv)
        plot_encode(df, args.output_dir)


if __name__ == "__main__":
    main()
