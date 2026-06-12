import argparse
from utils import TOKENIZER_CLASSES, load_tokenizer


def trace_word_bpe(word: str, merge_rules: list) -> None:
    """Print step-by-step BPE merge trace for a single word."""
    tokens = list(word) + ["</w>"]
    rules_index = {pair: i for i, pair in enumerate(merge_rules)}

    print(f"\nInitial: {tokens}\n")

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

    print(f"\nFinal: {tokens}")


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


def cmd_tokenize(args):
    tokenizer, _ = load_tokenizer(args.tokenizer, args.models_dir, args.vocab_size, args.n_train)
    tokens = tokenizer.encode(args.text)
    print(f"\nText   : {args.text}")
    print(f"Tokens : {tokens}")
    print(f"Count  : {len(tokens)} tokens")


def cmd_trace(args):
    tokenizer, _ = load_tokenizer(args.tokenizer, args.models_dir, args.vocab_size, args.n_train)
    is_wp = args.tokenizer.startswith("wp")
    if is_wp:
        print(f"\nTracing WordPiece merges for: {args.word!r}")
        trace_word_wp(args.word, tokenizer.merge_rules)
    else:
        print(f"\nTracing BPE merges for: {args.word!r}")
        trace_word_bpe(args.word, tokenizer.merge_rules)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo: tokenize a sentence or trace merge rules for a word.")
    parser.add_argument("--tokenizer",  type=str, required=True,
                        help=f"Tokenizer to use: {list(TOKENIZER_CLASSES.keys())}")
    parser.add_argument("--models-dir", type=str, default="results/models")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--n_train",    type=int, required=True)

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_tok = subparsers.add_parser("tokenize", help="Tokenize a sentence")
    p_tok.add_argument("--text", type=str, required=True, help="Sentence to tokenize")

    p_trace = subparsers.add_parser("trace", help="Trace merge rules for a single word")
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
