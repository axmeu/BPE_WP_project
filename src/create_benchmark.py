"""
Dataset created and pushed on HF, for informations only
(does not need to be run)

To load:
    from datasets import load_dataset
    load_dataset(axmeu/morphscore_fr)

    or with pandas:
    load_dataset(axmeu/morphscore_fr)["train"].to_pandas()

Based on: MorphScore (Arnett, Hudspeth & O'Connor, 2025)
"""
import argparse
import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import login


def parse_args():
    parser = argparse.ArgumentParser(description="Create morphological benchmark from MorphScore")
    parser.add_argument("--morphscore-id", type=str, default="catherinearnett/morphscore")
    parser.add_argument("--create-benchmark", action="store_true",
                        help="Build and push benchmark to HuggingFace")
    parser.add_argument("--hub-id",        type=str, default="axmeu/morphscore_fr")
    parser.add_argument("--hf-token",      type=str, default=None)
    return parser.parse_args()


def build_morphemes(row) -> list[str]:
    morphemes = []
    if pd.notna(row["preceding_part"]) and row["preceding_part"]:
        morphemes.append(row["preceding_part"])
    morphemes.append(row["stem"])
    if pd.notna(row["following_part"]) and row["following_part"]:
        morphemes.append(row["following_part"])
    return morphemes


def build_benchmark(morphscore_id: str) -> pd.DataFrame:
    print("Loading MorphScore...")
    df = load_dataset(morphscore_id)["train"].to_pandas()

    df_fr = df[(df["language"] == "fra_latn") &
               (df["pos"] == "VERB") &
               (df["unique"] == "unique")].copy()
    print(f"FR verbs unique : {len(df_fr)}")

    df_fr["morphemes"] = df_fr.apply(build_morphemes, axis=1)

    df_fr["valid"] = df_fr["morphemes"].apply("".join) == df_fr["wordform"].str.lower()
    df_fr = df_fr[df_fr["valid"]].copy()

    df_fr = df_fr.rename(columns={"wordform": "word"})
    benchmark = df_fr[["word", "morphemes"]].reset_index(drop=True)

    print(f"Benchmark size: {len(benchmark)}")
    return benchmark


def main():
    args = parse_args()

    if args.create_benchmark:
        benchmark = build_benchmark(args.morphscore_id)

        if args.hf_token:
            login(token=args.hf_token)

        Dataset.from_pandas(benchmark).push_to_hub(args.hub_id)
        print(f"Benchmark pushed to https://huggingface.co/datasets/{args.hub_id}")


if __name__ == "__main__":
    main()
