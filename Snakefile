configfile: "config.yaml"

N_SCALING_STR = ",".join([str(n) for n in config["n_scaling"]])
VOCAB_SCALING  = config["vocab_scaling"][0]
TOKENIZERS_STR = ",".join(config["tokenizers_fast"])

SCALING_TARGETS = expand(
    "results/models/{tokenizer}_v{vocab}_n{n}.json",
    tokenizer=config["tokenizers_scaling"],
    vocab=config["vocab_scaling"],
    n=config["n_scaling"],
) + ["results/plots/scaling.png"]

FULL_TARGETS = ["results/eval.csv"]

rule all:
    input:
        SCALING_TARGETS
        + FULL_TARGETS


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

rule plot_scaling:
    input:
        expand(
            "results/models/{tokenizer}_v{vocab}_n{n}_meta.json",
            tokenizer=config["tokenizers_scaling"],
            vocab=config["vocab_scaling"],
            n=config["n_scaling"],
        )
    output:
        "results/plots/scaling.png"
    shell:
        """
        pixi run python src/plot.py \
            --vocab_size {VOCAB_SCALING} \
            --n_scaling {N_SCALING_STR}
        """

rule evaluate:
    input:
        expand(
            "results/models/{tokenizer}_v{{vocab}}_n{{n}}.json",
            tokenizer=config["tokenizers_fast"],
        )
    output:
        temp("results/eval_{vocab}_{n}.csv")
    shell:
        """
        pixi run python src/evaluate.py \
            --tokenizers {TOKENIZERS_STR} \
            --vocab_size {wildcards.vocab} \
            --n_train {wildcards.n} \
            --hub-id {config[wikipedia_hub_id]} \
            --morpho-id {config[morpho_benchmark_id]} \
            --output {output} \
            --save_csv true
        """

rule merge_eval:
    input:
        expand(
            "results/eval_{vocab}_{n}.csv",
            vocab=config["vocab_sizes"],
            n=config["n_train"],
        )
    output:
        "results/eval.csv"
    run:
        import pandas as pd
        dfs = [pd.read_csv(f) for f in input]
        pd.concat(dfs, ignore_index=True).to_csv(output[0], index=False)