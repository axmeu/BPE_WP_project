import argparse
import csv
import json
import time
from pathlib import Path
from datasets import load_dataset
from BPE.naive import BPE
from BPE.fast import FastBPE
from WordPiece.naive import WordPiece
from WordPiece.fast import FastWordPiece


TOKENIZER_CLASSES = {
    "bpe_naive": BPE,
    "bpe_fast":  FastBPE,
    "wp_naive":  WordPiece,
    "wp_fast":   FastWordPiece
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tokenizers")
    parser.add_argument("--models-dir", type=str, default="results/models",
                        help="Directory containing trained model .json files")
    parser.add_argument("--tokenizers", type=str, default="all",
                        help="Tokenizers to evaluate: 'all' or comma-separated e.g. 'bpe_fast,wp_fast'")
    parser.add_argument("--vocab_size", type=int, required=True,
                        help="Vocab size of the models to load")
    parser.add_argument("--n_train",    type=int, required=True,
                        help="n_train of the models to load")
    parser.add_argument("--n_test",     type=int, default=None,
                        help="Number of test articles (default: all)")
    parser.add_argument("--hub-id",     type=str, default="axmeu/BPE_WP_dataset",
                        help="HuggingFace dataset ID")
    parser.add_argument("--output",     type=str, default="results/eval.csv",
                        help="Output CSV path")
    return parser.parse_args()


def load_tokenizer(name: str, models_dir: str, vocab_size: int, n_train: int):
    model_name = f"{name}_v{vocab_size}_n{n_train}"
    path = Path(models_dir) / f"{model_name}.json"
    meta_path = Path(models_dir) / f"{model_name}_meta.json"

    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    tokenizer = TOKENIZER_CLASSES[name].load(str(path))

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return tokenizer, meta


def encode_time(tokenizer, texts: list[str]) -> tuple[list, float]:
    start = time.time()
    encoded = [tokenizer.encode(t) for t in texts]
    return encoded, time.time() - start


def compression(tokenizer, texts: list[str]) -> float:
    # |corpus characters| / |test set|
    for text in texts:
        ...


def subwords_per_words(tokenizer, texts: list[str]) -> tuple[float, float]:  # mean, std
    # |tokens| / |words| (mean at sentence level)
    ...


def main():
    args = parse_args()

    if args.tokenizers == "all":
        tokenizer_names = list(TOKENIZER_CLASSES.keys())
    else:
        tokenizer_names = args.tokenizers.split(",")

    print("Loading dataset...")
    ds = load_dataset(args.hub_id)
    test_texts = ds["test"]["text"][:args.n_test]
    print(f"Test: {len(test_texts)} articles")

    results = []

    for name in tokenizer_names:
        print(f"\nLoading {name}...")
        try:
            tokenizer, meta = load_tokenizer(name, args.models_dir, args.vocab_size, args.n_train)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue

        print(f"Encoding {name}...")
        encoded, enc_time = encode_time(tokenizer, test_texts)
        print(f"  encode_time: {enc_time:.3f}s")

        results.append({
            "model":        name,
            "vocab_size":   args.vocab_size,
            "n_train":      args.n_train,
            "n_test":       len(test_texts),
            "train_time":   meta.get("train_time", None),
            "encode_time":  round(enc_time, 3)
        })

    file_exists = Path(args.output).exists()
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
