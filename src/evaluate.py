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
import numpy as np
import regex
import pandas as pd


TOKENIZER_CLASSES = {
    "bpe_naive": BPE,
    "bpe_fast":  FastBPE,
    "wp_naive":  WordPiece,
    "wp_fast":   FastWordPiece
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate tokenizers")
    parser.add_argument("--models-dir",  type=str, default="results/models")
    parser.add_argument("--tokenizers",  type=str, default="all")
    parser.add_argument("--vocab_size",  type=int, required=True)
    parser.add_argument("--n_train",     type=int, required=True)
    parser.add_argument("--n_test",      type=int, default=None)
    parser.add_argument("--hub-id",      type=str, default="axmeu/wiki_fr")
    parser.add_argument("--morpho-id",   type=str, default="axmeu/morphscore_fr")
    parser.add_argument("--output",      type=str, default="results/eval.csv")
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


def compression(encoded: list[list[str]], texts: list[str]) -> float:
    total_char = sum(len(text) for text in texts)
    total_tokens = sum(len(tokens) for tokens in encoded)
    return total_char / total_tokens


def fertility(encoded: list[list[str]], texts: list[str]) -> tuple[float, float]:
    fertilities = []
    for tokens, text in zip(encoded, texts):
        words = regex.findall(r"[\p{L}\p{N}]+", text)
        if words:
            fertilities.append(len(tokens) / len(words))
    return float(np.mean(fertilities)), float(np.std(fertilities))


def clean_tokens(tokens: list[str]) -> list[str]:
    return [t.replace("</w>", "").replace("##", "") for t in tokens if t not in ("</w>", "##")]


def get_in_vocab_words(tokenizer, words: list[str]) -> set[str]:
    return {word for word in words if len(tokenizer.encode(word)) == 1}


def precision_recall_f1(tokens: list[str], gold: list[str]) -> tuple[float, float, float]:
    clean = set(clean_tokens(tokens))
    gold_s = set(gold)
    tp = len(clean & gold_s)
    precision = tp / len(clean) if clean else 0.0
    recall = tp / len(gold_s) if gold_s else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def boundary_f1(tokens: list[str], gold: list[str]) -> float:
    def to_boundaries(segments: list[str]) -> set[int]:
        boundaries = set()
        pos = 0
        for seg in segments[:-1]:
            pos += len(seg)
            boundaries.add(pos)
        return boundaries

    pred_boundaries = to_boundaries(clean_tokens(tokens))
    gold_boundaries = to_boundaries(gold)

    tp = len(pred_boundaries & gold_boundaries)
    fp = len(pred_boundaries - gold_boundaries)
    fn = len(gold_boundaries - pred_boundaries)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def eval_morpho(tokenizer, df: pd.DataFrame) -> dict:
    df = df.copy()
    df["in_vocab"] = df["word"].str.lower().apply(lambda w: len(tokenizer.encode(w)) == 1)

    print(f"in_vocab:  {df['in_vocab'].sum()}")
    print(f"out_vocab: {(~df['in_vocab']).sum()}")

    records = []
    for _, row in df.iterrows():
        word = row["word"]
        gold = row["morphemes"]
        tokens = tokenizer.encode(word)

        p, r, f1_m = precision_recall_f1(tokens, gold)
        b_f1 = boundary_f1(tokens, gold)

        records.append({
            "in_vocab":    row["in_vocab"],
            "precision":   p,
            "recall":      r,
            "f1_morpheme": f1_m,
            "f1_boundary": b_f1
        })

    results_df = pd.DataFrame(records)
    out_vocab = results_df[~results_df["in_vocab"]]

    def agg(subset):
        return {
            "precision":   round(subset["precision"].mean(),   4),
            "recall":      round(subset["recall"].mean(),      4),
            "f1_morpheme": round(subset["f1_morpheme"].mean(), 4),
            "f1_boundary": round(subset["f1_boundary"].mean(), 4),
            "n":           len(subset)
        }

    global_metrics = agg(results_df)
    out_vocab_metrics = agg(out_vocab)

    print(f"Global: P={global_metrics['precision']} R={global_metrics['recall']} "
          f"F1_m={global_metrics['f1_morpheme']} F1_b={global_metrics['f1_boundary']}")
    print(f"out_vocab: F1_b={out_vocab_metrics['f1_boundary']} (n={out_vocab_metrics['n']})")

    return {
        "morpho_global":    global_metrics,
        "morpho_out_vocab": out_vocab_metrics
    }


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

    print("Loading morphological benchmark...")
    morpho_df = load_dataset(args.morpho_id)["train"].to_pandas()
    print(f"Benchmark: {len(morpho_df)} words")

    results = []

    for name in tokenizer_names:
        print(f"\n{'='*50}")
        print(f"Evaluating {name}...")
        try:
            tokenizer, meta = load_tokenizer(name, args.models_dir, args.vocab_size, args.n_train)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue

        encoded, enc_time = encode_time(tokenizer, test_texts)
        comp = compression(encoded, test_texts)
        fert_mean, fert_std = fertility(encoded, test_texts)

        print(f"encode_time:      {enc_time:.3f}s")
        print(f"compression:      {comp:.3f}")
        print(f"fertility (mean): {fert_mean:.3f} ± {fert_std:.3f}")

        print("Running morphological evaluation...")
        morpho = eval_morpho(tokenizer, morpho_df)

        row = {"model":                  name,
               "vocab_size":             args.vocab_size,
               "n_train":                args.n_train,
               "n_test":                 len(test_texts),
               "train_time":             meta.get("train_time"),
               "encode_time":            round(enc_time, 3),
               "compression":            round(comp, 4),
               "fertility_mean":         round(fert_mean, 4),
               "fertility_std":          round(fert_std, 4),
               "morpho_precision":       morpho["morpho_global"]["precision"],
               "morpho_recall":          morpho["morpho_global"]["recall"],
               "morpho_f1_morpheme":     morpho["morpho_global"]["f1_morpheme"],
               "morpho_f1_boundary":     morpho["morpho_global"]["f1_boundary"],
               "morpho_n":               morpho["morpho_global"]["n"],
               "morpho_out_vocab_f1_b":  morpho["morpho_out_vocab"]["f1_boundary"],
               "morpho_out_vocab_n":     morpho["morpho_out_vocab"]["n"]}
        results.append(row)

    if not results:
        print("No results to save.")
        return

    file_exists = Path(args.output).exists()
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
