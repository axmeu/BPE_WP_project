import regex
from collections import defaultdict
import json
from pathlib import Path


class WordPiece:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merge_rules = []

    def _pretokenize(self, texts: list[str]) -> dict:
        word_freqs = defaultdict(int)
        for text in texts:
            for word in regex.findall(r"[a-zA-ZÀ-ÿŒœ]+", text):
                word_freqs[tuple([word[0]] + [f"##{c}" for c in word[1:]])] += 1
        return word_freqs

    def _get_symbol_freqs(self, word_freqs: dict) -> dict:
        symbol_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            for symbol in word:
                symbol_freqs[symbol] += freq
        return symbol_freqs

    def _get_pair_scores(self, word_freqs: dict) -> tuple[dict, dict]:
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_freqs[(word[i], word[i + 1])] += freq

        symbol_freqs = self._get_symbol_freqs(word_freqs)

        pair_scores = {}
        for pair, freq in pair_freqs.items():
            denom = symbol_freqs[pair[0]] * symbol_freqs[pair[1]]
            pair_scores[pair] = freq / denom if denom > 0 else 0.0

        return pair_scores, pair_freqs

    def _merge_pair(self, pair: tuple, word_freqs: dict) -> dict:
        merged_symbol = pair[0] + (pair[1][2:] if pair[1].startswith("##") else pair[1])
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(merged_symbol)
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
            pair_scores, _ = self._get_pair_scores(word_freqs)
            if not pair_scores:
                break

            best_pair = max(pair_scores, key=pair_scores.get)
            word_freqs = self._merge_pair(best_pair, word_freqs)

            merged_symbol = best_pair[0] + (
                best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[1]
            )
            self.vocab.add(merged_symbol)
            self.merge_rules.append(best_pair)
            iteration += 1

            if verbose and iteration % 1000 == 0:
                print(f"{iteration} merges... vocab size: {len(self.vocab)}")

        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")

    def encode(self, text: str, unk_token: str = "[UNK]") -> list[str]:
        tokens = []
        for unit in regex.findall(r"[a-zA-ZÀ-ÿŒœ]+|[^a-zA-ZÀ-ÿŒœ\s]", text):
            if not regex.match(r"[a-zA-ZÀ-ÿŒœ]+", unit):
                tokens.append(unit)
                continue
            chars = list(unit)
            start = 0
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    single = chars[start] if start == 0 else "##" + chars[start]
                    cur_substr = single if single in self.vocab else unk_token
                    end = start + 1
                tokens.append(cur_substr)
                start = end
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
    def load(cls, path: str) -> "WordPiece":
        with open(path) as f:
            data = json.load(f)
        tokenizer = cls(data["vocab_size"])
        tokenizer.vocab = set(data["vocab"])
        tokenizer.merge_rules = [tuple(pair) for pair in data["merge_rules"]]
        return tokenizer
