configfile: "config.yaml"

N_SCALING_STR  = ",".join([str(n) for n in config["n_scaling"]])
VOCAB_SCALING  = config["vocab_scaling"][0]
TOKENIZERS_STR = ",".join(config["tokenizers_fast"])

SCALING_TARGETS = expand(
    "results/models/{tokenizer}_v{vocab}_n{n}.json",
    tokenizer=config["tokenizers_scaling"],
    vocab=config["vocab_scaling"],
    n=config["n_scaling"],
) + ["results/plots/scaling.png"]

FULL_TARGETS = ["results/eval.csv", 
                "results/morpho_ex.csv",
                "results/plots/morpho.png",
                "results/plots/encode.png"]

rule all:
    input:
        SCALING_TARGETS
        + FULL_TARGETS

def get_min_frequency(wildcards):
    if int(wildcards.vocab) == config["vocab_scaling"][0]:
        return config["min_frequency_scaling"]
    return config["min_frequency"]

rule train:
    output:
        rules = "results/models/{tokenizer}_v{vocab}_n{n}.json",
        meta  = "results/models/{tokenizer}_v{vocab}_n{n}_meta.json",
    params:
        min_freq = get_min_frequency
    shell:
        """
        pixi run python src/train.py \
            --tokenizer {wildcards.tokenizer} \
            --vocab-size {wildcards.vocab} \
            --n_train {wildcards.n} \
            --hub_id {config[wikipedia_hub_id]} \
            --min_frequency {params.min_freq}
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
        pixi run python src/plot.py scaling \
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
        eval_csv      = temp("results/eval_{vocab}_{n}.csv"),
        morpho_ex_csv = temp("results/morpho_ex_{vocab}_{n}.csv")
    shell:
        """
        pixi run python src/evaluate.py \
            --tokenizers {TOKENIZERS_STR} \
            --vocab_size {wildcards.vocab} \
            --n_train {wildcards.n} \
            --hub-id {config[wikipedia_hub_id]} \
            --morpho-id {config[morpho_benchmark_id]} \
            --output {output.eval_csv} \
            --ex_output {output.morpho_ex_csv} \
            --save_csv true
        """

rule evaluate_baseline:
    output:
        eval_csv      = "results/eval_camembert.csv",
        morpho_ex_csv = "results/morpho_ex_camembert.csv"
    shell:
        """
        pixi run python src/evaluate.py \
            --tokenizers camembert \
            --vocab_size 0 \
            --n_train 0 \
            --hub-id {config[wikipedia_hub_id]} \
            --morpho-id {config[morpho_benchmark_id]} \
            --output {output.eval_csv} \
            --ex_output {output.morpho_ex_csv} \
            --save_csv true
        """

rule merge_eval:
    input:
        expand(
            "results/eval_{vocab}_{n}.csv",
            vocab=config["vocab_sizes"],
            n=config["n_train"],
        ) + ["results/eval_camembert.csv"]
    output:
        "results/eval.csv"
    run:
        import pandas as pd
        dfs = [pd.read_csv(f) for f in input]
        pd.concat(dfs, ignore_index=True).to_csv(output[0], index=False)

rule merge_morpho_ex:
    input:
        expand(
            "results/morpho_ex_{vocab}_{n}.csv",
            vocab=config["vocab_sizes"],
            n=config["n_train"]
        ) + ["results/morpho_ex_camembert.csv"]
    output:
        "results/morpho_ex.csv"
    run:
        import pandas as pd
        dfs = [pd.read_csv(f) for f in input]
        pd.concat(dfs, ignore_index=True).to_csv(output[0], index=False)

rule plot_morpho:
    input:
        "results/eval.csv"
    output:
        "results/plots/morpho.png"
    shell:
        """
        pixi run python src/plot.py morpho --eval-csv {input}
        """

rule plot_encode:
    input:
        "results/eval.csv"
    output:
        "results/plots/encode.png"
    shell:
        """
        pixi run python src/plot.py encode --eval-csv {input}
        """