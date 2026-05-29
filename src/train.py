import argparse
import json
import time
import sys
from pathlib import Path
from datasets import load_dataset
from BPE.naive import BPE
from BPE.fast import FastBPE
from WordPiece.naive import WordPiece
import subprocess
import platform
# from WordPiece.fast import FastWordPiece


TOKENIZERS = {
    "bpe_naive": lambda vocab_size: BPE(vocab_size),
    "bpe_fast": lambda vocab_size: FastBPE(vocab_size),
    "wp_naive": lambda vocab_size: WordPiece(vocab_size),
    # "wp_fast": lambda vocab_size: FastWordPiece(vocab_size),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer and save merge rules")
    parser.add_argument("--tokenizer",  type=str, required=True,
                        help=f"Tokenizer to train: {list(TOKENIZERS.keys())}")
    parser.add_argument("--vocab-size", type=int, default=20_000,
                        help="Vocabulary size (default: 20 000)")
    parser.add_argument("--n_train",    type=int, default=None,
                        help="Number of train articles (default: all)")
    parser.add_argument("--hub_id",     type=str, default="axmeu/BPE_WP_dataset",
                        help="HuggingFace dataset ID")
    parser.add_argument("--output-dir", type=str, default="results/models",
                        help="Directory to save model and metadata (default: results/models)")
    return parser.parse_args()


def main():
    cpu = subprocess.check_output("lscpu | grep 'Model name' | cut -d: -f2", shell=True)\
        .decode().strip()
        
    args = parse_args()

    if args.tokenizer not in TOKENIZERS:
        print(f"Unknown tokenizer '{args.tokenizer}'. Choose from: {list(TOKENIZERS.keys())}")
        sys.exit(1)

    print("Loading dataset...")
    ds = load_dataset(args.hub_id)
    train_texts = ds["train"]["text"][:args.n_train]
    print(f"Train: {len(train_texts)} articles")

    print(f"Training {args.tokenizer}...")
    tokenizer = TOKENIZERS[args.tokenizer](args.vocab_size)
    start = time.time()
    tokenizer.train(train_texts)
    train_time = time.time() - start
    print(f"Done in {train_time:.3f}s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    model_name = f"{args.tokenizer}_v{args.vocab_size}_n{len(train_texts)}"
    rules_path = output_dir / f"{model_name}.json"
    meta_path = output_dir / f"{model_name}_meta.json"

    tokenizer.save(str(rules_path))

    with open(meta_path, "w") as f:
        json.dump({
            "tokenizer":  args.tokenizer,
            "vocab_size": args.vocab_size,
            "n_train":    len(train_texts),
            "train_time": round(train_time, 3),
            "cpu":        cpu,
            "python":     platform.python_version()
        }, f, indent=2)

    print(f"Model saved to {rules_path}")
    print(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
