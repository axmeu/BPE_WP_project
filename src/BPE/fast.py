import regex
import heapq
from collections import defaultdict
import json
from pathlib import Path


class FastBPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = set()
        self.merge_rules = []
        self._cache = {}

    def _pretokenize(self, texts: list[str]) -> tuple[dict, dict, int]:
        raw_freqs = defaultdict(int)
        for text in texts:
            for word in regex.findall(r"[\p{L}\p{N}]+", text):
                raw_freqs[word] += 1

        word_tokens = {}
        word_freqs = {}
        for word_id, (word, freq) in enumerate(raw_freqs.items()):
            word_tokens[word_id] = list(word) + ["</w>"]
            word_freqs[word_id] = freq

        return word_tokens, word_freqs, len(raw_freqs)

    def _build_index(
        self,
        word_tokens: dict,
        word_freqs: dict,
    ) -> tuple[dict, list]:

        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)

        for word_id, tokens in word_tokens.items():
            freq = word_freqs[word_id]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(word_id)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        return pair_counts, pair_to_words, heap

    def _merge_pair_fast(
        self,
        best_pair: tuple,
        word_tokens: dict,
        word_freqs: dict,
        pair_counts: dict,
        pair_to_words: dict,
        heap: list,
    ) -> None:

        merged = best_pair[0] + best_pair[1]
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

            for i, token in enumerate(new_tokens):
                if token == merged:
                    if i > 0:
                        left_pair = (new_tokens[i - 1], merged)
                        pair_counts[left_pair] += freq
                        pair_to_words[left_pair].add(word_id)
                        heapq.heappush(heap, (-pair_counts[left_pair], left_pair))

                    if i < len(new_tokens) - 1:
                        right_pair = (merged, new_tokens[i + 1])
                        pair_counts[right_pair] += freq
                        pair_to_words[right_pair].add(word_id)
                        heapq.heappush(heap, (-pair_counts[right_pair], right_pair))

        del pair_to_words[best_pair]

    def train(self, texts: list[str], verbose: bool = True) -> None:
        word_tokens, word_freqs, _ = self._pretokenize(texts)

        self.vocab = set(sym for tokens in word_tokens.values() for sym in tokens)

        if verbose:
            print(f"Initial vocab size: {len(self.vocab)}")
            print(f"Unique words: {len(word_tokens):,}")

        pair_counts, pair_to_words, heap = self._build_index(word_tokens, word_freqs)

        iteration = 0

        while len(self.vocab) < self.vocab_size:
            best_pair = None
            while heap:
                neg_count, pair = heapq.heappop(heap)
                real_count = pair_counts.get(pair, 0)
                if real_count > 0 and -neg_count == real_count:
                    best_pair = pair
                    break

            if best_pair is None:
                break

            self._merge_pair_fast(
                best_pair, word_tokens, word_freqs,
                pair_counts, pair_to_words, heap
            )

            merged = best_pair[0] + best_pair[1]
            self.vocab.add(merged)
            self.merge_rules.append(best_pair)
            iteration += 1

            if verbose and iteration % 1000 == 0:
                print(f"{iteration} merges... vocab size: {len(self.vocab)}")

        if verbose:
            print(f"Final vocab size: {len(self.vocab)}")

        self._rules_index = {pair: i for i, pair in enumerate(self.merge_rules)}

    def _encode_word(self, word: str) -> list[str]:
        if word + "</w>" in self.vocab:
            return [word + "</w>"]

        word_tokens = list(word) + ["</w>"]
        while len(word_tokens) > 1:
            best_idx = None
            best_pos = None
            for i in range(len(word_tokens) - 1):
                rank = self._rules_index.get((word_tokens[i], word_tokens[i + 1]))
                if rank is not None and (best_idx is None or rank < best_idx):
                    best_idx = rank
                    best_pos = i
            if best_pos is None:
                break
            word_tokens = word_tokens[:best_pos] \
                + [word_tokens[best_pos] + word_tokens[best_pos + 1]] + word_tokens[best_pos + 2:]
        return word_tokens

    def encode(self, text: str) -> list[str]:
        tokens = []
        for word in regex.findall(r"[\p{L}\p{N}]+", text):
            if word not in self._cache:
                self._cache[word] = self._encode_word(word)
            tokens.extend(self._cache[word])
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
    def load(cls, path: str) -> "FastBPE":
        with open(path) as f:
            data = json.load(f)
        tokenizer = cls(data["vocab_size"])
        tokenizer.vocab = set(data["vocab"])
        tokenizer.merge_rules = [tuple(pair) for pair in data["merge_rules"]]
        tokenizer._rules_index = {pair: i for i, pair in enumerate(tokenizer.merge_rules)}
        return tokenizer
