"""
Dataset created and pushed on HF, for informations only
(does not need to be run)

To load:
    from datasets import load_dataset
    import pandas as pd
    df = load_dataset("axmeu/wiki_fr")["train"].to_pandas()

Train: 300k articles | Test: 5k articles
"""

import argparse
import json
import hashlib
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login


N_TRAIN = 300_000
N_TEST = 5_000
N_TOTAL = N_TRAIN + N_TEST


def parse_args():
    parser = argparse.ArgumentParser(description="Create and upload French Wikipedia dataset to HuggingFace Hub")
    parser.add_argument("--hub-id",  type=str, required=True,
                        help="HuggingFace dataset ID, e.g. username/wikipedia-fr-tokenization")
    parser.add_argument("--n-train", type=int, default=N_TRAIN,
                        help=f"Number of train articles (default: {N_TRAIN})")
    parser.add_argument("--n-test",  type=int, default=N_TEST,
                        help=f"Number of test articles (default: {N_TEST})")
    parser.add_argument("--hf-token", type=str, required=True,
                        help="HuggingFace token")
    return parser.parse_args()


def extract_samples(n: int) -> list[str]:
    print("Loading French Wikipedia in streaming mode...")
    raw = load_dataset(
        "wikimedia/wikipedia", "20231101.fr", split="train", streaming=True
    )

    samples = []
    for i, example in enumerate(raw):
        if i >= n:
            break
        samples.append(example["text"])
        if i % 10_000 == 0:
            print(f"  {i}/{n} articles extracted")

    return samples


def main():
    args = parse_args()
    n_total = args.n_train + args.n_test

    login(token=args.hf_token)

    samples = extract_samples(n_total)

    print("Pushing to HuggingFace Hub...")
    DatasetDict({
        "train": Dataset.from_dict({"text": samples[:args.n_train]}),
        "test":  Dataset.from_dict({"text": samples[args.n_train:]})
    }).push_to_hub(args.hub_id)

    print(f"Dataset available at: https://huggingface.co/datasets/{args.hub_id}")


if __name__ == "__main__":
    main()
