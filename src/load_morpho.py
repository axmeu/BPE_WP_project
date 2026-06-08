import argparse
import io
import re
import zipfile
import requests
import pandas as pd
from pathlib import Path
from datasets import Dataset
from huggingface_hub import login


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Lexique4 and create morphological benchmark")
    parser.add_argument("--url",              type=str,
                        default="http://www.lexique.org/databases/Lexique400/Lexique400.zip")
    parser.add_argument("--directory",        type=str, default="data/lexique")
    parser.add_argument("--lexique-file",     type=str, default="Lexique4/Lexique4.tsv")
    parser.add_argument("--create-benchmark", action="store_true",
                        help="Build and push benchmark to HuggingFace")
    parser.add_argument("--hub-id",           type=str, default="axmeu/FrVMorpho")
    parser.add_argument("--hf-token",         type=str, default=None)
    parser.add_argument("--n-per-structure",  type=int, default=100)
    return parser.parse_args()


def download_and_extract(url: str, directory: str) -> Path:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(path)
    print(f"Extracted to {path}")
    return path


def parse_morph(decomp: str) -> list[str]:
    clean = decomp.replace("_", "/").replace("{", "").replace("}", "")
    morphemes = re.split(r"[/.]", clean)
    morphemes = [re.sub(r"\(.*?\)", "", m).strip() for m in morphemes]
    return [m for m in morphemes if m]


def build_benchmark(lexique_path: Path, n: int) -> pd.DataFrame:
    df = pd.read_csv(lexique_path, sep="\t",
                     usecols=["1_Mot", "5_Cgram", "31_MorphoStruct", "32_MorphoDecomp"])
    df = df.dropna(subset=["1_Mot", "5_Cgram", "31_MorphoStruct", "32_MorphoDecomp"])
    df = df[df["5_Cgram"] == "VER"]
    df = df[~df["32_MorphoDecomp"].str.contains(r"\[", regex=True)]

    df["morphemes"] = df.apply(lambda r: parse_morph(r["32_MorphoDecomp"]), axis=1)
    df = df[df.apply(lambda r: "".join(r["morphemes"]) == r["1_Mot"].lower(), axis=1)]

    benchmark = (df[df["31_MorphoStruct"] != "0-1-0"]
                 .groupby("31_MorphoStruct")
                 .filter(lambda x: len(x) >= n)
                 .groupby("31_MorphoStruct")
                 .apply(lambda x: x.sample(min(len(x), n), random_state=42))
                 .reset_index(level=0)
                 .reset_index(drop=True))

    print(f"Benchmark size: {len(benchmark)}")
    print(benchmark["31_MorphoStruct"].value_counts())
    return benchmark


def main():
    args = parse_args()

    path = download_and_extract(args.url, args.directory)

    if args.create_benchmark:
        lexique_path = path / args.lexique_file
        benchmark = build_benchmark(lexique_path, args.n_per_structure)

        if args.hf_token:
            login(token=args.hf_token)

        Dataset.from_pandas(benchmark).push_to_hub(args.hub_id)
        print(f"Benchmark pushed to https://huggingface.co/datasets/{args.hub_id}")


if __name__ == "__main__":
    main()
