import argparse
from pathlib import Path

from BPE.naive import BPE
from BPE.fast import FastBPE
from WordPiece.naive import WordPiece
from WordPiece.fast import FastWordPiece


TOKENIZER_CLASSES = {
    "bpe_naive": BPE,
    "bpe_fast":  FastBPE,
    "wp_naive":  WordPiece,
    "wp_fast":   FastWordPiece,
}


# helpers 

def load_tokenizer(tokenizer_name: str, model_path: str):
    if tokenizer_name not in TOKENIZER_CLASSES:
        raise ValueError(
            f"Unknown tokenizer '{tokenizer_name}'. "
            f"Choose from: {list(TOKENIZER_CLASSES.keys())}"
        )
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return TOKENIZER_CLASSES[tokenizer_name].load(str(path))


def trace_word_bpe(word: str, merge_rules: list) -> None:
    """Print step-by-step BPE merge trace for a single word."""
    tokens = list(word) + ["</w>"]
    rules_index = {pair: i for i, pair in enumerate(merge_rules)}

    print(f"\nInitial : {tokens}\n")

    step = 0
    while len(tokens) > 1:
        best_idx = None
        best_pos = None
        for i in range(len(tokens) - 1):
            rank = rules_index.get((tokens[i], tokens[i + 1]))
            if rank is not None and (best_idx is None or rank < best_idx):
                best_idx = rank
                best_pos = i

        if best_pos is None:
            break

        merged = tokens[best_pos] + tokens[best_pos + 1]
        print(
            f"  Step {step + 1:<4} (rule #{best_idx:<6}): "
            f"{tokens[best_pos]!r:<12} + {tokens[best_pos + 1]!r:<12} → {merged!r}"
        )
        tokens = tokens[:best_pos] + [merged] + tokens[best_pos + 2:]
        step += 1

    print(f"\nFinal   : {tokens}")


def trace_word_wp(word: str, merge_rules: list) -> None:
    """Print step-by-step WordPiece merge trace for a single word."""
    tokens = [word[0]] + [f"##{c}" for c in word[1:]]

    def merged_symbol(pair):
        a, b = pair
        return a + (b[2:] if b.startswith("##") else b)

    print(f"\nInitial : {tokens}\n")

    step = 0
    for pair in merge_rules:
        i = 0
        changed = False
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i + 1]) == pair:
                m = merged_symbol(pair)
                if not changed:
                    print(
                        f"  Step {step + 1:<4}: "
                        f"{tokens[i]!r:<12} + {tokens[i + 1]!r:<12} → {m!r}"
                    )
                    changed = True
                tokens = tokens[:i] + [m] + tokens[i + 2:]
            else:
                i += 1
        if changed:
            step += 1

    print(f"\nFinal   : {tokens}")


# subcommands 

def cmd_tokenize(args):
    print(f"Loading {args.tokenizer} from {args.model} ...")
    tokenizer = load_tokenizer(args.tokenizer, args.model)

    tokens = tokenizer.encode(args.text)

    print(f"\nText   : {args.text}")
    print(f"Tokens : {tokens}")
    print(f"Count  : {len(tokens)} tokens")


def cmd_trace(args):
    is_wp = args.tokenizer.startswith("wp")

    if is_wp:
        print(f"Loading {args.tokenizer} from {args.model} ...")
        tokenizer = load_tokenizer(args.tokenizer, args.model)
        print(f"\nTracing WordPiece merges for: {args.word!r}")
        trace_word_wp(args.word, tokenizer.merge_rules)
    else:
        print(f"Loading {args.tokenizer} from {args.model} ...")
        tokenizer = load_tokenizer(args.tokenizer, args.model)
        print(f"\nTracing BPE merges for: {args.word!r}")
        trace_word_bpe(args.word, tokenizer.merge_rules)


# CLI 

def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo: tokenize a sentence or trace merge rules for a word.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True,
        help=f"Tokenizer to use: {list(TOKENIZER_CLASSES.keys())}",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to a trained model .json file (e.g. results/models/bpe_fast_v32000_n180000.json)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_tok = subparsers.add_parser("tokenize", help="Tokenize a sentence")
    p_tok.add_argument("--text", type=str, required=True, help="Sentence to tokenize")

    p_trace = subparsers.add_parser("trace", help="Trace merge rules for a single word (BPE or WordPiece)")
    p_trace.add_argument("--word", type=str, required=True, help="Word to trace")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "tokenize":
        cmd_tokenize(args)
    elif args.command == "trace":
        cmd_trace(args)


if __name__ == "__main__":
    main()
