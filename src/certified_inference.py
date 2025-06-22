import collections
import os
from typing import Any, Dict, Tuple
import json

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import wandb

import utils

from utils.data import (
    get_dataset_config_name,
    TaskDatasetPrompting,
    PubMedQADataset,
    SQuADDataset,
)

cluster = utils.cluster.ClusterManager()
model_dir = os.path.join(cluster.artifact_dir, "models")


def get_data(cfg, tokenizer: AutoTokenizer):
    split = cfg.data.split
    N = {"test": cfg.data.test_size, "val": cfg.data.val_size, "train": cfg.data.train_size}[split]

    if cfg.data.name == "task_data":
        kwargs = {
            "modality": cfg.data.modality,
            "task": cfg.data.task,
            "max_seq_length": cfg.data.max_sequence_length,
            "num_int": cfg.data.num_int,
            "num_char": cfg.data.num_char,
            "constant_seq_length": cfg.data.constant_sequence_length,
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "TaskData"),
        }

        dataset = TaskDatasetPrompting(N=N, split=split, **kwargs)

    elif cfg.data.name == "pubmedqa":
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "PubMedQA"),
            "return_sequence": cfg.data.return_sequence,
            "debug": cfg.data.debug,
            "truncate_sequence_to": cfg.data.truncate_sequence_to,
            # "overwrite_tokenizer_name": cfg.data.overwrite_tokenizer_name,
        }
        dataset = PubMedQADataset(N=N, split=split, **kwargs)

    elif cfg.data.name == "squad":
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "SQuAD"),
            "return_sequence": cfg.data.return_sequence,
            "debug": cfg.data.debug,
            "categories": cfg.data.categories,
            "truncate_sequence_to": cfg.data.truncate_sequence_to,
            "min_response_length": cfg.data.min_response_length,
            "sequence_prefix": None if cfg.data.sequence_prefix in [None, "None", "none", ""] else cfg.data.sequence_prefix,
            # "overwrite_tokenizer_name": cfg.data.overwrite_tokenizer_name,
        }
        dataset = SQuADDataset(N=N, split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset {cfg.data.name}")

    return dataset


def get_model(cfg: DictConfig, tokenizer_model: AutoTokenizer, tokenizer_generator: AutoTokenizer) -> utils.MetaModel:
    model_cache_dir = os.path.join(cluster.artifact_dir, "models")

    # model
    if cfg.model.source == "hf":
        model_name_or_path = cfg.model.name_or_path
    elif cfg.model.source == "local":
        model_name_or_path = os.path.join(cluster.artifact_dir, "models", cfg.model.name_or_path)
    else:
        raise ValueError(f"Unknown model source {cfg.model.source}")

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map=cfg.model.device, cache_dir=model_cache_dir)
    model = utils.models.set_model_precision(model, cfg.model.precision)
    model_generation_config = GenerationConfig(
        max_new_tokens=cfg.model.generation_max_new_tokens,  # max_new_tokens
        temperature=cfg.model.temperature,
        top_k=cfg.model.top_k if cfg.model.top_k > 0 else None,
        do_sample=True,
        num_return_sequences=1,
        renomalize_logits=True,
        pad_token_id=tokenizer_model.pad_token_id,
        eos_token_id=tokenizer_model.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )

    if cfg.model.tokenizer.add_pad_token:
        assert tokenizer_model.pad_token_id is not None
        model.resize_token_embeddings(len(tokenizer_model))

    # generator
    if cfg.generator.source == "hf":
        generator_name_or_path = cfg.generator.name_or_path
    elif cfg.generator.source == "local":
        generator_name_or_path = os.path.join(cluster.artifact_dir, "models", cfg.generator.name_or_path)
    else:
        raise ValueError(f"Unknown generator source {cfg.generator.source}")

    generator = AutoModelForCausalLM.from_pretrained(generator_name_or_path, device_map=cfg.generator.device, cache_dir=model_cache_dir)
    generator = utils.models.set_model_precision(generator, cfg.generator.precision)

    generator_config = OmegaConf.create({"temperature": cfg.generator.temperature, "top_k": cfg.generator.top_k})

    if cfg.generator.tokenizer.add_pad_token:
        assert tokenizer_generator.pad_token_id is not None
        generator.resize_token_embeddings(len(tokenizer_generator))

    # meta model
    meta_model = utils.MetaModel(
        tokenizer_model=tokenizer_model,
        tokenizer_generator=tokenizer_generator,
        model=model,
        model_generation_config=model_generation_config,
        generator=generator,
        generator_config=generator_config,
        k=cfg.meta_model.k,
        T=cfg.meta_model.T,
        distribution_model=cfg.meta_model.distribution_model,
        distribution_generator=cfg.meta_model.distribution_generator,
        divergence=cfg.meta_model.divergence,
    )

    return meta_model


def get_judge(cfg: DictConfig, run) -> utils.judge.Judge:
    return utils.judge.Judge(
        cfg.judge.name_or_path,
        cfg.judge.template,
        cfg.judge.device,
        cfg.judge.precision,
        cluster,
        decision_strategy=cfg.judge.decision_strategy,
        regression_model_path=cfg.judge.regression_model_path,
    )


def setup_logging(cfg: DictConfig) -> Tuple[Any, str, str]:
    logging_mode = cfg.log.mode
    if cfg.run.debug:
        logging_mode = "disabled"
    if cluster.network == "offline" and logging_mode == "online":
        logging_mode = "offline"

    run = wandb.init(
        entity="<YOUR_WANDB_ENTITY>",
        project="<YOUR_WANDB_PROJECT>",
        config=OmegaConf.to_container(cfg),
        job_type="certify",
        tags=cfg.log.tags,
        dir=cluster.log_dir,
        mode=logging_mode,
    )

    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    return run, run.name


def compute_confusion_matrix(abstained, judge_decision) -> Dict[str, int]:
    """
    Compute the confusion matrix between abstained (True/False) and judge_decision (IN, OUT, NOT-NL, NA).

    Parameters:
    - abstained (list of bool): List indicating if the judgement was abstained.
    - judge_decision (list of str): Corresponding decisions by the judge.

    Returns:
    - dict: Dictionary with tuple keys representing (abstained, decision) combinations and their counts.
    """
    # Initialize the confusion matrix with all possible combinations
    decision_categories = ["IN", "OUT", "INVALID"]
    matrix = {(abstain, dec): 0 for abstain in [True, False] for dec in decision_categories}

    # Populate the confusion matrix with counts
    for abstain, decision in zip(abstained, judge_decision):
        matrix[(abstain, decision)] += 1

    matrix = {f"conf_matrix_abstained={abstain}-decision={decision}": count for (abstain, decision), count in matrix.items()}
    matrix = {k: v / len(judge_decision) for k, v in matrix.items()}

    return matrix


def validate_cfg(cfg: DictConfig):
    if cfg.judge.enabled:
        raise NotImplementedError("Judge is not implemented yet.")


@hydra.main(config_path="config", config_name="certified_inference", version_base="1.3")
def main(cfg: DictConfig):
    validate_cfg(cfg)

    utils.seed_everything(cfg.run.seed)

    cfg.data_config_name = get_dataset_config_name(cfg)

    run, run_name = setup_logging(cfg)

    # tokenizer
    tokenizer_model, tokenizer_generator = utils.tokenizers.get_tokenizers(cfg, model_dir)

    # data
    test_dataset = get_data(cfg, tokenizer_model)

    # model
    model = get_model(cfg, tokenizer_model, tokenizer_generator)

    # judge
    if cfg.judge.enabled:
        judge = get_judge(cfg, run)

    results_list = []

    # certify
    for batch in tqdm(test_dataset, desc="Certifying Inference", smoothing=0.05):
        # input_ids, target_ids = batch["input_ids"].unsqueeze(0), batch["target_ids"].unsqueeze(0)
        input_ids = batch["input_ids"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        n_prompt_length = batch["n_token_prompt"].item()
        prompt_ids = input_ids[:, :n_prompt_length]
        target_ids = input_ids[:, n_prompt_length:]

        output = model.generate(prompt_ids)
        output_ids = output.sequences[-1][0]

        # decode the texts
        decoded_prompt = tokenizer_model.decode(prompt_ids[0], skip_special_tokens=True)
        decoded_output = tokenizer_model.decode(output_ids, skip_special_tokens=True)
        decoded_response = tokenizer_model.decode(output_ids[len(prompt_ids[0]) :], skip_special_tokens=True)
        decoded_target = tokenizer_model.decode(target_ids[0], skip_special_tokens=True)

        if cfg.judge.enabled:
            judge_output_response = judge(decoded_response, cfg.topic)
            judge_output_target = judge(decoded_target, cfg.topic)
            judge_results = {
                "decision": judge_output_response.decision.value,
                "judge_output": judge_output_response.to_dict(),
                "decision_target": judge_output_target.decision.value,
                "judge_output_target": judge_output_target.to_dict(),
            }
        else:
            judge_results = {"decision": "NA", "judge_output": None, "decision_target": "NA", "judge_output_target": None}

        # gather outputs
        results_list.append(
            {
                "abstained": output.abstained,
                "num_samples": output.num_samples,
                "log_prob_f": output.log_probs,
                "log_prob_g": output.log_probs_g,
                "log_ratio": output.log_ratios,
                "upper_bounds": output.upper_bounds,
                "prompt_text": decoded_prompt,
                "generated_text": decoded_response,
                "target_text": decoded_target,
                **judge_results,
            }
        )

    # gather metrics
    abstained = [result["abstained"] for result in results_list]
    num_samples = [result["num_samples"] for result in results_list]
    num_samples_not_abstained = [num_samples[i] for i, abstain in enumerate(abstained) if not abstain]

    # log on wandb
    metrics = {
        "mean_abstained": np.mean(abstained),
        "mean_num_samples": np.mean(num_samples),
        "mean_num_samples_not_abstained": np.mean(num_samples_not_abstained),
    }

    if cfg.judge.enabled:
        judge_decision = [result["decision"] for result in results_list]
        judge_decision_counts = {"IN": 0, "OUT": 0, "INVALID": 0} | collections.Counter(judge_decision)
        judge_decision_proportions = {k: v / len(judge_decision) for k, v in judge_decision_counts.items()}
        confusion_matrix = compute_confusion_matrix(abstained, judge_decision)
        metrics |= {
            **{f"prop_judge_{k}": v for k, v in judge_decision_proportions.items()},
            **confusion_matrix,
        }

    run.log(metrics)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"\t{key}: {value}")

    # save the results
    all = {
        "config": OmegaConf.to_container(cfg),
        "results": results_list,
        "metrics": metrics,
    }
    save_path = os.path.join(cluster.artifact_dir, "certified_inference", run_name, "results.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all, f)
    print(f"Results saved at {save_path}")


if __name__ == "__main__":
    main()
