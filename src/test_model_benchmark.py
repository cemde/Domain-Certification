# ruff: noqa: E402
import os
import pickle
from typing import Any, Dict, List

import rich
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig

from utils.data import MMLUScoringDataset, mmlu_categories, PubMedQAScoringDataset, MMLU_SCORING_MARKER, PUBMEDQA_SCORING_MARKER
import utils

torch.set_printoptions(linewidth=220)
cluster = utils.cluster.ClusterManager()


def get_data(cfg, tokenizer: AutoTokenizer):
    if cfg.data.name == "mmlu" and cfg.data.variant == "scoring":
        if cfg.data.truncate_sequence_to in ["None", "NONE", "none", ""]:
            truncate_sequence_to = None
        else:
            truncate_sequence_to = int(cfg.data.truncate_sequence_to)

        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "MMLU"),
            "debug": cfg.data.debug,
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "truncate_sequence_to": truncate_sequence_to,
            "use_chat_template": cfg.data.use_chat_template,
            "categories": mmlu_categories(cfg.data.categories),
            "n_shot": cfg.data.n_shot,
        }

        test_dataset = MMLUScoringDataset(N=cfg.data.test_size, split="test", **kwargs)
        scoring_marker = MMLU_SCORING_MARKER

    elif cfg.data.name == "pubmedqa" and cfg.data.variant == "scoring":
        if cfg.data.truncate_sequence_to in ["None", "NONE", "none", ""]:
            truncate_sequence_to = None
        else:
            truncate_sequence_to = int(cfg.data.truncate_sequence_to)

        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "PubMedQA"),
            "debug": cfg.data.debug,
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "truncate_sequence_to": truncate_sequence_to,
            "use_chat_template": cfg.data.use_chat_template,
        }

        test_dataset = PubMedQAScoringDataset(N=cfg.data.test_size, split="test", **kwargs)
        scoring_marker = PUBMEDQA_SCORING_MARKER

    return test_dataset, scoring_marker


def get_model(cfg: DictConfig, tokenizer: AutoTokenizer):
    model_name_or_path = {"hf": cfg.model.name_or_path, "local": os.path.join(model_dir, cfg.model.name_or_path)}[cfg.model.source]
    try:
        kwargs = {"config": AutoConfig.from_pretrained(model_name_or_path)}
    except OSError:
        print(f"Could not load Model AutoConfig from {model_name_or_path}.")
        kwargs = {}
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", cache_dir=model_dir, **kwargs)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model={cfg.model.name_or_path}. Size: {model_size:,} ({model_size / (1000**2):.2f}M) parameters.")

    return model


def generate_text(
    generation_config: GenerationConfig, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_ids: torch.Tensor
) -> Dict[str, Any]:
    # Generate a response
    model.eval()

    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    assert prompt_ids.dim() == 2

    with torch.inference_mode():
        # Generate output sequence
        outputs = model.generate(
            inputs=prompt_ids,
            generation_config=generation_config,
        )

    return outputs


def subset_prompt(batch: List[Dict[str, torch.Tensor]], prompt_length: str) -> List[Dict[str, torch.Tensor]]:
    prompts = []
    for i, element in enumerate(batch):
        if prompt_length == "dataset":
            n_token_prompt = element["n_token_prompt"][0]
        else:
            n_token_prompt = int(prompt_length)
        prompts.append(element["input_ids"][:n_token_prompt])

    return prompts


def validate_cfg(cfg: DictConfig):
    if cfg.data.name not in ["mmlu", "pubmedqa"]:
        raise ValueError(f"Invalid data.name: {cfg.data.name}")

    if cfg.data.variant not in ["scoring"]:
        raise ValueError(f"Invalid data.variant: {cfg.data.variant}")

    if not cfg.data.return_sequence == "full":
        raise ValueError(
            f"Benchmarking requires the dataset to return full sequences. data.return_Sequence='full'. Got: {cfg.data.return_sequence} instead."
        )

    if not cfg.log.save_results:
        rich.print("[bold yellow]Results will not be saved. Set log.save_results=True to save results.[/bold yellow]")


@hydra.main(config_path="config", config_name="benchmark", version_base="1.3")
def main(cfg: DictConfig):
    SAVE_DIR = os.path.join(cluster.artifact_dir, "benchmarks", cfg.data.name, cfg.model.name_or_path.replace("/", "_"))
    TEST_CONFIG = f"categories-{cfg.data.categories}-{cfg.data.n_shot}-shot"
    os.makedirs(SAVE_DIR, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(SAVE_DIR, "config.yaml"))
    rich.print(f"[bold green]Using Directory: {SAVE_DIR} for predictions.[/bold green]")
    utils.seed_everything(cfg.run.seed)

    # validate settings
    validate_cfg(cfg)

    # tokenizer
    model_name_or_path = {"hf": cfg.model.name_or_path, "local": os.path.join(model_dir, cfg.model.name_or_path)}[cfg.model.source]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=model_dir)

    # get data
    test_dataset, scoring_marker = get_data(cfg, tokenizer)
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # get model
    model = get_model(cfg, tokenizer)

    # inference
    answer_tokens = [tokenizer.encode(answer_marker, add_special_tokens=False)[0] for answer_marker in scoring_marker]
    results = []
    for idx, batch in enumerate(pbar := tqdm(test_dataset, desc="Benchmark", unit="questions")):
        if batch["input_ids"].dim() == 1:
            batch["input_ids"] = batch["input_ids"].unsqueeze(0)
        input_ids = batch["input_ids"].to(model.device)
        correct_answer_id = batch["correct_answer_id"].item()

        with torch.inference_mode():
            model_output = model(input_ids=input_ids)

        answer_logits = model_output.logits[:, -1, answer_tokens].squeeze()
        answer_id = torch.argmax(answer_logits, dim=-1).item()

        results.append(
            {
                **utils.move_tree_to_device(batch, "cpu"),
                "correct_answer_id": correct_answer_id,
                "answer_id": answer_id,
                "answer_logits": answer_logits.cpu(),
                "correct": int(answer_id == correct_answer_id),
            }
        )

    # score
    df = pd.DataFrame(results)[["idx", "subject", "answer_id", "correct_answer_id", "correct"]]
    # convert idx to int
    df["idx"] = df["idx"].apply(lambda x: int(x.item()))
    correctness_by_subject = df.groupby("subject")["correct"].mean()
    overall_accuracy = df["correct"].mean()

    # print results
    print(f"Benchmark Dataset: {cfg.data.name}")
    print(f"Model: {cfg.model.name_or_path}")
    print(f"Accuracy: {overall_accuracy:.2%}")
    print("Subject Accuracy:")
    for subject, accuracy in correctness_by_subject.items():
        print(f"  {subject}: {accuracy:.2%}")

    # save results
    all_results = {
        "results": results,
        "config": cfg,
        "metrics": {
            "overall_accuracy": overall_accuracy,
            "correctness_by_subject": correctness_by_subject.to_dict(),
        },
    }

    # save predictions
    if cfg.log.save_results:
        results_path = os.path.join(SAVE_DIR, f"results-{TEST_CONFIG}.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    model_dir = os.path.join(cluster.artifact_dir, "models")

    main()

    print("Done!")
