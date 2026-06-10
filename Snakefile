# Snakefile
configfile: "config.yaml"

rule all:
    input:
        expand(
            "results/models/{tokenizer}_v{vocab}_n{n}_meta.json",
            tokenizer=config["tokenizers"],
            vocab=config["vocab_sizes"],
            n=config["n_train"],
        )

rule train:
    output:
        rules = "results/models/{tokenizer}_v{vocab}_n{n}.json",
        meta  = "results/models/{tokenizer}_v{vocab}_n{n}_meta.json",
    shell:
        """
        pixi run python src/train.py \
            --tokenizer {wildcards.tokenizer} \
            --vocab-size {wildcards.vocab} \
            --n_train {wildcards.n} \
            --hub_id {config[wikipedia_hub_id]}
        """