import regex
from collections import defaultdict
from pathlib import Path
import json


class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merge_rules = []

    def _pretokenize(self, texts: list[str]) -> dict:
        word_freqs = defaultdict(int)
        for text in texts:
            for word in regex.findall(r"[a-zA-ZÀ-ÿ0-9]+", text):
                word_freqs[tuple(list(word) + ["</w>"])] += 1
        return word_freqs

    def _get_pair_freqs(self, word_freqs: dict) -> dict:
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += freq
        return pair_freqs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, texts: list[str], verbose: bool = True) -> None:
        word_freqs = self._pretokenize(texts)

        self.vocab = set(sym for word in word_freqs for sym in word)

        if verbose:
            print(f"Initial vocab size: {len(self.vocab)}")
            print(f"Unique words: {len(word_freqs):,}")

        iteration = 0

        while len(self.vocab) < self.vocab_size:
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)

            new_symbol = best_pair[0] + best_pair[1]
            self.vocab.add(new_symbol)
            self.merge_rules.append(best_pair)
            iteration += 1

            if verbose and iteration % 1000 == 0:
                print(f"{iteration} merges... vocab size: {len(self.vocab)}")

        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")

    def encode(self, text: str) -> list[str]:
        tokens = []
        for unit in regex.findall(r"[a-zA-ZÀ-ÿ0-9]+|[^a-zA-ZÀ-ÿ0-9\s]", text):
            if regex.match(r"[a-zA-ZÀ-ÿ0-9]+", unit):
                word_tokens = list(unit) + ["</w>"]
                for pair in self.merge_rules:
                    i = 0
                    while i < len(word_tokens) - 1:
                        if (word_tokens[i], word_tokens[i + 1]) == pair:
                            word_tokens = word_tokens[:i] + [word_tokens[i] + word_tokens[i + 1]] \
                                        + word_tokens[i + 2:]
                        else:
                            i += 1
                tokens.extend(word_tokens)
            else:
                tokens.append(unit)
        return tokens

    def save(self, path: str) -> None:
        data = {
            "vocab_size":   self.vocab_size,
            "vocab":        list(self.vocab),
            "merge_rules":  [list(pair) for pair in self.merge_rules]
        }
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPE":
        with open(path) as f:
            data = json.load(f)
        tokenizer = cls(data["vocab_size"])
        tokenizer.vocab = set(data["vocab"])
        tokenizer.merge_rules = [tuple(pair) for pair in data["merge_rules"]]
        return tokenizer
