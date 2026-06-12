import regex
import heapq
import json
from collections import defaultdict
from pathlib import Path


class FastWordPiece:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merge_rules = []

    def _pretokenize(self, texts: list[str]) -> tuple[dict, dict]:
        raw_freqs = defaultdict(int)
        for text in texts:
            for word in regex.findall(r"[a-zA-ZÀ-ÿ0-9]+", text):
                raw_freqs[word] += 1

        word_tokens = {}
        word_freqs = {}
        for word_id, (word, freq) in enumerate(raw_freqs.items()):
            word_tokens[word_id] = [word[0]] + [f"##{c}" for c in word[1:]]
            word_freqs[word_id] = freq

        return word_tokens, word_freqs

    def _build_index(self, word_tokens: dict, word_freqs: dict) -> tuple:
        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)
        symbol_counts = defaultdict(int)

        for word_id, tokens in word_tokens.items():
            freq = word_freqs[word_id]
            for symbol in tokens:
                symbol_counts[symbol] += freq
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(word_id)

        heap = []
        for pair, count in pair_counts.items():
            denom = symbol_counts[pair[0]] * symbol_counts[pair[1]]
            score = count / denom if denom > 0 else 0.0
            heapq.heappush(heap, (-score, pair))

        return pair_counts, pair_to_words, symbol_counts, heap

    @staticmethod
    def _merged_symbol(pair: tuple) -> str:
        a, b = pair
        return a + (b[2:] if b.startswith("##") else b)

    def _merge_pair_fast(
        self,
        best_pair: tuple,
        word_tokens: dict,
        word_freqs: dict,
        pair_counts: dict,
        pair_to_words: dict,
        symbol_counts: dict,
        heap: list,
    ) -> None:
        merged = self._merged_symbol(best_pair)
        affected_words = list(pair_to_words[best_pair])

        for word_id in affected_words:
            tokens = word_tokens[word_id]
            freq = word_freqs[word_id]

            i = 0
            new_tokens = []

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                    if i > 0:
                        left_pair = (tokens[i - 1], tokens[i])
                        pair_counts[left_pair] -= freq
                        pair_to_words[left_pair].discard(word_id)

                    if i < len(tokens) - 2:
                        right_pair = (tokens[i + 1], tokens[i + 2])
                        pair_counts[right_pair] -= freq
                        pair_to_words[right_pair].discard(word_id)

                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            word_tokens[word_id] = new_tokens

            n_merges = new_tokens.count(merged)
            symbol_counts[best_pair[0]] -= freq * n_merges
            symbol_counts[best_pair[1]] -= freq * n_merges
            symbol_counts[merged] += freq * n_merges

            for i, token in enumerate(new_tokens):
                if token == merged:
                    if i > 0:
                        left_pair = (new_tokens[i - 1], merged)
                        pair_counts[left_pair] += freq
                        pair_to_words[left_pair].add(word_id)
                        denom = symbol_counts[left_pair[0]] * symbol_counts[left_pair[1]]
                        score = pair_counts[left_pair] / denom if denom > 0 else 0.0
                        heapq.heappush(heap, (-score, left_pair))

                    if i < len(new_tokens) - 1:
                        right_pair = (merged, new_tokens[i + 1])
                        pair_counts[right_pair] += freq
                        pair_to_words[right_pair].add(word_id)
                        denom = symbol_counts[right_pair[0]] * symbol_counts[right_pair[1]]
                        score = pair_counts[right_pair] / denom if denom > 0 else 0.0
                        heapq.heappush(heap, (-score, right_pair))

        del pair_to_words[best_pair]

    def train(self, texts: list[str], verbose: bool = True) -> None:
        word_tokens, word_freqs = self._pretokenize(texts)
        self.vocab = set(sym for tokens in word_tokens.values() for sym in tokens)

        if verbose:
            print(f"Initial vocab size: {len(self.vocab)}")
            print(f"Unique words: {len(word_tokens):,}")

        pair_counts, pair_to_words, symbol_counts, heap = self._build_index(word_tokens, word_freqs)

        iteration = 0

        while len(self.vocab) < self.vocab_size:
            best_pair = None
            while heap:
                neg_score, pair = heapq.heappop(heap)
                count = pair_counts.get(pair, 0)
                if count > 0:
                    denom = symbol_counts[pair[0]] * symbol_counts[pair[1]]
                    real_score = count / denom if denom > 0 else 0.0
                    if abs(-neg_score - real_score) < 1e-10:
                        best_pair = pair
                        break

            if best_pair is None:
                break

            self._merge_pair_fast(
                best_pair, word_tokens, word_freqs,
                pair_counts, pair_to_words, symbol_counts, heap
            )

            self.vocab.add(self._merged_symbol(best_pair))
            self.merge_rules.append(best_pair)
            iteration += 1

            if verbose and iteration % 1000 == 0:
                print(f"{iteration} merges... vocab size: {len(self.vocab)}")

        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")

    def encode(self, text: str, unk_token: str = "[UNK]") -> list[str]:
        tokens = []
        for unit in regex.findall(r"[a-zA-ZÀ-ÿ0-9]+|[^a-zA-ZÀ-ÿ0-9\s]", text):
            if not regex.match(r"[a-zA-ZÀ-ÿ0-9]+", unit):
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
    def load(cls, path: str) -> "FastWordPiece":
        with open(path) as f:
            data = json.load(f)
        tokenizer = cls(data["vocab_size"])
        tokenizer.vocab = set(data["vocab"])
        tokenizer.merge_rules = [tuple(pair) for pair in data["merge_rules"]]
        return tokenizer
