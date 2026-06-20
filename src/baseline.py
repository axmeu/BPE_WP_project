from transformers import AutoTokenizer


class CamembertWrapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("camembert-base")
        self.vocab = set(self.tokenizer.get_vocab().keys())

    def encode(self, text: str) -> list[str]:
        return self.tokenizer.tokenize(text)
