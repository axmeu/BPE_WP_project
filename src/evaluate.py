import argparse
import csv
import time
from pathlib import Path
from datasets import load_dataset
import numpy as np
import regex
import pandas as pd
from utils import TOKENIZER_CLASSES, load_tokenizer


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
    parser.add_argument("--save_csv",    type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--n_examples",  type=int, default=3)
    parser.add_argument("--ex_output",   type=str, default=None)
    return parser.parse_args()


def encode_time(tokenizer, texts: list[str]) -> tuple[list, float]:
    start = time.time()
    encoded = [tokenizer.encode(t) for t in texts]
    return encoded, time.time() - start


def compression(encoded: list[list[str]], texts: list[str]) -> tuple[float, int]:
    total_char = sum(len(text) for text in texts)
    total_tokens = sum(len(tokens) for tokens in encoded)
    compression_ratio = total_char / total_tokens
    return compression_ratio, total_tokens


def fertility(encoded: list[list[str]], texts: list[str]) -> tuple[float, float]:
    fertilities = []
    for tokens, text in zip(encoded, texts):
        words = regex.findall(r"[a-zA-ZÀ-ÿŒœ]+", text)
        if words:
            fertilities.append(len(tokens) / len(words))
    return float(np.mean(fertilities)), float(np.std(fertilities))


def clean_tokens(tokens: list[str]) -> list[str]:
    return [t.replace("</w>", "").replace("##", "") for t in tokens if t not in ("</w>", "##")]


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
    records = []
    for _, row in df.iterrows():
        word = row["word"]
        gold = row["morphemes"]
        tokens = tokenizer.encode(word)

        p, r, f1_m = precision_recall_f1(tokens, gold)
        b_f1 = boundary_f1(tokens, gold)

        records.append({
            "precision":   p,
            "recall":      r,
            "f1_morpheme": f1_m,
            "f1_boundary": b_f1
        })

    results_df = pd.DataFrame(records)

    metrics = {
        "morpho_precision":   round(results_df["precision"].mean(),   4),
        "morpho_recall":      round(results_df["recall"].mean(),      4),
        "morpho_f1_morpheme": round(results_df["f1_morpheme"].mean(), 4),
        "morpho_f1_boundary": round(results_df["f1_boundary"].mean(), 4),
        "morpho_n":           len(results_df)
    }

    print(f"P={metrics['morpho_precision']} R={metrics['morpho_recall']} "
          f"F1_m={metrics['morpho_f1_morpheme']} F1_b={metrics['morpho_f1_boundary']} "
          f"(n={metrics['morpho_n']})")

    return metrics


def sample_examples(all_tokenizers: dict, common_df: pd.DataFrame, n_examples: int = 3,
                    seed: int = 42) -> list[dict]:
    sample = common_df.sample(n_examples, random_state=seed)
    rows = []
    for _, row in sample.iterrows():
        word = row["word"]
        gold = row["morphemes"]

        def to_boundaries(segments):
            boundaries, pos = [], 0
            for seg in segments[:-1]:
                pos += len(seg)
                boundaries.append(pos)
            return boundaries

        record = {
            "word": word,
            "gold": " + ".join(gold),
            "gold_boundaries": str(to_boundaries(gold)),
        }

        for name, (tokenizer, _) in all_tokenizers.items():
            tokens = tokenizer.encode(word)
            clean = clean_tokens(tokens)

            p, r, f1_m = precision_recall_f1(tokens, gold)
            b_f1 = boundary_f1(tokens, gold)

            record[name] = " + ".join(clean)
            record[f"{name}_boundaries"] = str(to_boundaries(clean))
            record[f"{name}_f1_morph"] = round(f1_m, 3)
            record[f"{name}_f1_bound"] = round(b_f1, 3)

        rows.append(record)
    return rows


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

    all_tokenizers = {}
    for name in tokenizer_names:
        try:
            tokenizer, meta = load_tokenizer(name, args.models_dir, args.vocab_size, args.n_train)
            all_tokenizers[name] = (tokenizer, meta)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")

    if not all_tokenizers:
        print("No tokenizers loaded.")
        return

    words = morpho_df["word"].str.lower().tolist()
    out_vocab_sets = {
        name: {w for w in words if len(t.encode(w)) > 1}
        for name, (t, _) in all_tokenizers.items()
    }

    if len(out_vocab_sets) > 1:
        common = set.intersection(*out_vocab_sets.values())
        print(f"\nCommon out_vocab: {len(common)} words "
              f"(intersection of {len(all_tokenizers)} tokenizers)")
    else:
        common = list(out_vocab_sets.values())[0]
        print(f"\nout_vocab: {len(common)} words")

    common_df = morpho_df[morpho_df["word"].str.lower().isin(common)].reset_index(drop=True)

    if args.ex_output:
        examples = sample_examples(all_tokenizers, common_df, args.n_examples)
        examples_df = pd.DataFrame(examples)
        examples_df.insert(0, "vocab_size", args.vocab_size)
        file_exists = Path(args.ex_output).exists()
        examples_df.to_csv(args.ex_output, mode="a", header=not file_exists, index=False)
        print(f"\n{examples_df.to_string(index=False)}")
        print(f"Examples saved to {args.ex_output}")

    results = []

    for name, (tokenizer, meta) in all_tokenizers.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {name}...")

        encoded, enc_time = encode_time(tokenizer, test_texts)
        compression_ratio, total_tokens = compression(encoded, test_texts)
        fert_mean, fert_std = fertility(encoded, test_texts)

        print(f"  encode_time:          {enc_time:.3f}s")
        print(f"  total tokens:         {total_tokens}")
        print(f"  compression:          {compression_ratio:.3f}")
        print(f"  fertility:            {fert_mean:.3f} ± {fert_std:.3f}")

        print("Running morphological evaluation...")
        morpho = eval_morpho(tokenizer, common_df)

        row = {"model":                name,
               "vocab_size":           args.vocab_size,
               "n_train":              args.n_train,
               "n_test":               len(test_texts),
               "train_time":           meta.get("train_time"),
               "encode_time":          round(enc_time, 3),
               "total_encoded_tokens": total_tokens,
               "compression":          round(compression_ratio, 4),
               "fertility_mean":       round(fert_mean, 4),
               "fertility_std":        round(fert_std, 4),
               **morpho}
        results.append(row)

    if not results:
        print("No results to save.")
        return

    if args.save_csv:
        file_exists = Path(args.output).exists()
        with open(args.output, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.output}")
    else:
        print("\nCSV save skipped.")


if __name__ == "__main__":
    main()
