# BPE & WordPiece Tokenization

Implementation of BPE and WordPiece tokenization algorithms from scratch in Python, with naive and fast versions. Trained on French Wikipedia (180k articles), evaluated on training and encoding time, compression metrics and morphological analysis

## Requirements
Only requires Pixi, which handles all dependencies for Linux and macOS.
- [Pixi](https://pixi.sh)

## Installation

```bash
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/axmeu/BPE_WP_project.git
cd BPE_WP_project
pixi install
```

## Development

To use the Pixi environment as a Jupyter kernel:

```bash
pixi run python -m ipykernel install --user --name pixi-env --display-name "pixi-env"
```

Then select `pixi-env` as the kernel in Jupyter.

## Usage

**Full pipeline:**
Train only for now, adjust `--cores` based on available memory:
```bash
pixi run snakemake --cores 4
```

**Train a specific tokenizer:**
```bash
pixi run python src/train.py --help
```
Example:
```bash
pixi run python src/train.py --tokenizer bpe_fast --vocab-size 20000 --n_train 10000
```

**Evaluate:**
- Train & encoding time
- Vocabulary compression metrics
- Morphological analysis (to be implemented)
```bash
pixi run python src/evaluate.py --help
```
Example:
```bash
pixi run python src/evaluate.py --tokenizers bpe_fast,wp_fast --vocab-size 20000 --n-train 10000 --n-test 1000
```

**Quick demo:**
(To be implemented)
- Tokenize the given sentence
- Trace rules for a given word (function present in notebook)


## Datasets

- French Wikipedia: `axmeu/wiki_fr`
- Morphological benchmark: `axmeu/morphscore_fr`

## References
(To be added)