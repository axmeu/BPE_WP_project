from pathlib import Path
import json
from BPE.naive import BPE
from BPE.fast import FastBPE
from WordPiece.naive import WordPiece
from WordPiece.fast import FastWordPiece


TOKENIZER_CLASSES = {
    "bpe_naive": BPE,
    "bpe_fast":  FastBPE,
    "wp_naive":  WordPiece,
    "wp_fast":   FastWordPiece
}


def load_tokenizer(name: str, models_dir: str, vocab_size: int, n_train: int):
    if name not in TOKENIZER_CLASSES:
        raise ValueError(f"Unknown tokenizer '{name}'. Choose: {list(TOKENIZER_CLASSES.keys())}")
    model_name = f"{name}_v{vocab_size}_n{n_train}"
    path = Path(models_dir) / f"{model_name}.json"
    meta_path = Path(models_dir) / f"{model_name}_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    tokenizer = TOKENIZER_CLASSES[name].load(str(path))
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    return tokenizer, meta
