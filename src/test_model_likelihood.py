# ruff: noqa: E402
import collections
import os
import pickle
from typing import Any, Dict

import rich
from tqdm import tqdm
import utils.cluster
import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
)

from utils.data import TaskDatasetTraining, TaskDatasetPrompting, TaskCollectionDataset
import utils


torch.set_printoptions(linewidth=220)
cluster = utils.cluster.ClusterManager()


def get_data(cfg, tokenizer: AutoTokenizer):
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

        test_dataset = TaskDatasetPrompting(N=cfg.data.test_size, split="test", **kwargs)

    elif cfg.data.name == "task_data_collection":
        kwargs = {
            "modality": cfg.data.modality,
            "max_seq_length": cfg.data.max_sequence_length,
            "num_int": cfg.data.num_int,
            "num_char": cfg.data.num_char,
            "constant_seq_length": cfg.data.constant_sequence_length,
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "TaskData"),
        }

        test_datasets = []

        for task in cfg.data.task.split(","):
            kwargs["task"] = task
            test_datasets.append(TaskDatasetTraining(N=cfg.data.test_size, split="test", **kwargs))

        test_dataset = TaskCollectionDataset(N=cfg.data.test_size, datasets=test_datasets)

    # depending on the data, set the max length of the entire sequence
    max_length = cfg.model.generation_max_length

    # print some examples
    # utils.data.print_examples(test_dataset, "test", tokenizer)
    return test_dataset, max_length


def get_model(cfg: DictConfig, tokenizer: AutoTokenizer):
    model_path = os.path.join(model_dir, cfg.model.name_or_path)
    try:
        kwargs = {"config": AutoConfig.from_pretrained(model_path)}
    except OSError:
        print(f"Could not load Model AutoConfig from {model_path}.")
        kwargs = {}
    model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=model_dir, **kwargs)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model={cfg.model.name_or_path}. Size: {model_size:,} ({model_size / (1000**2):.2f}M) parameters.")

    return model


def token_log_likelihood(
    tokenizer: AutoTokenizer, token_ids: torch.Tensor, logits: torch.Tensor, temperature: float = 1.0, auto_remove_eos: bool = True
):
    # get log likelihood of give sentence for all tokens that are generated.

    assert token_ids.dim() == 2, f"sentence should be 2D, got {token_ids.dim()}. batch x tokens"
    assert logits.dim() == 3, f"logits should be 3D, got {logits.dim()}. batch x tokens x vocab"
    assert token_ids.shape[0] == logits.shape[0], "batch size mismatch"
    assert token_ids.shape[1] == logits.shape[1], "sequence length mismatch"

    # remove EOS token
    if auto_remove_eos and token_ids[:, -1].eq(tokenizer.eos_token_id).all():
        # print(f"Removing EOS token for |{tokenizer.decode(token_ids[0], skip_special_token=False)}|")
        logits = logits[:, :-1, :]
        token_ids = token_ids[:, :-1]

    # Get the log probabilities of the response_ids
    logsoftmax = F.log_softmax(logits / temperature, dim=-1)
    log_likelihoods = torch.gather(logsoftmax, 2, token_ids.unsqueeze(-1)).squeeze(-1)
    return log_likelihoods


def generate_text(
    generation_config: GenerationConfig, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_ids: torch.Tensor
) -> Dict[str, Any]:
    # Generate a response
    model.eval()

    assert prompt_ids.dim() == 1
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    with torch.inference_mode():
        # Generate output sequence
        outputs = model.generate(
            inputs=prompt_ids,
            generation_config=generation_config,
        )

    rich.print("[bold red]Generate Text function only supports top_k=all, temperature=1.0[/bold red]")
    scores = torch.stack(outputs.logits, dim=1)
    prompt_length = prompt_ids.shape[1]
    responses = outputs.sequences[:, prompt_length:]

    # log_likelihood
    log_likelihood_token = token_log_likelihood(tokenizer, responses, scores, 1.0)
    log_likelihood = log_likelihood_token.sum(dim=1)

    # Convert prompt and output IDs to text
    prompt_text = [tokenizer.decode(p, skip_special_tokens=False) for p in prompt_ids]
    output_text = [tokenizer.decode(o, skip_special_tokens=False) for o in outputs.sequences]

    return {
        "prompt_ids": prompt_ids.cpu(),
        "output_ids": outputs.sequences.cpu(),
        "prompt_text": prompt_text,
        "output_text": output_text,
        "logits": scores.cpu(),
        "log_likelihood": log_likelihood.cpu(),
    }


def calculate_log_likelihood(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt_ids: torch.Tensor, sentence: torch.Tensor, temperature: float = 1.0
):
    # Use the model to get the logits for each token

    # reshape
    assert sentence.dim() == 1
    sentence = sentence.cuda().unsqueeze(0)
    assert sentence.dim() == 2

    # forward pass to get logits
    with torch.no_grad():
        outputs = model(sentence, labels=sentence)

    # remove prompt to get log p(y|x)
    prompt_length = prompt_ids.shape[1]
    response = sentence[:, prompt_length:]
    logits = outputs.logits[:, (prompt_length - 1) : -1]

    log_likelihoods = token_log_likelihood(tokenizer, response, logits, temperature)

    return log_likelihoods.sum().item()


@hydra.main(config_path="config", config_name="test_likelihood", version_base="1.3")
def main(cfg: DictConfig):
    data_config = f"{cfg.data.name}_{cfg.data.modality}_{cfg.data.task}"
    PREDICTION_DIR = os.path.join(cluster.artifact_dir, "predictions", data_config, cfg.model.name_or_path)
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(PREDICTION_DIR, "config.yaml"))
    rich.print(f"[bold green]Using Directory: {PREDICTION_DIR} for predictions.[/bold green]")
    # set seed
    utils.seed_everything(cfg.run.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, cfg.model.name_or_path), cache_dir=model_dir, padding_side="left")
    # tokenizer.pad_token = tokenizer.eos_token

    # data
    test_dataset, MAX_LENGTH = get_data(cfg, tokenizer)

    # get model
    model = get_model(cfg, tokenizer).cuda()

    generation_config = GenerationConfig(
        max_length=MAX_LENGTH,
        temperature=cfg.inference.temperature,
        top_k=cfg.inference.top_k,
        do_sample=cfg.inference.sample,
        num_return_sequences=cfg.inference.predict_n_sequences,
        renomalize_logits=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )

    preds = []
    for idx, batch in enumerate(tqdm(test_dataset, desc="Inference")):
        batch = {k: v.cuda() for k, v in batch.items() if isinstance(v, torch.Tensor)}
        generate_return = generate_text(generation_config, model, tokenizer, batch["input_ids"][:-1])  # remove EOS token from input_ids
        preds.append(
            generate_return
            | {"target_ids": batch["target_ids"], "target_text": tokenizer.decode(batch["target_ids"], skip_special_tokens=False)}
        )

    # Calculate log likelihood
    log_likelihoods = []
    for pred in preds:
        log_likelihood = calculate_log_likelihood(model, tokenizer, pred["prompt_ids"], pred["output_ids"][0], cfg.inference.temperature)
        log_likelihoods.append(log_likelihood)
        print(f"During Generation: {pred['log_likelihood'][0].item():.16f} Post Hoc: {log_likelihood:.16f}")

    log_likelihoods_target = []
    for pred in preds:
        log_likelihood = calculate_log_likelihood(model, tokenizer, pred["prompt_ids"], pred["target_ids"], cfg.inference.temperature)
        log_likelihoods_target.append(log_likelihood)
        print(f"During Generation: {pred['log_likelihood'][0].item():.16f} Post Hoc: {log_likelihood:.16f}")

    # print predicted and check for errors with target
    n_incorrects = []
    print("Outputs")
    for pred in preds:
        pred_ids = pred["output_ids"][0].cpu()
        target_ids = pred["target_ids"].cpu()
        # pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        # target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        special_token_ids = torch.tensor(tokenizer.all_special_ids)
        preds_ids_no_special = pred_ids[~torch.isin(pred_ids, special_token_ids)]
        target_ids_no_special = target_ids[~torch.isin(target_ids, special_token_ids)]

        min_len = min(preds_ids_no_special.shape[0], target_ids_no_special.shape[0])
        diff_len = abs(preds_ids_no_special.shape[0] - target_ids_no_special.shape[0])
        n_incorrect = (~(preds_ids_no_special[:min_len] == target_ids_no_special[:min_len])).sum(-1)

        if n_incorrect.ndim > 1:
            n_incorrect = n_incorrect[0]

        n_incorrect += diff_len
        n_incorrects.append(n_incorrect.item())

        # switch case 0 bold green, 1-2 bold yellow, 3+ bold red
        match n_incorrect:
            case 0:
                color = "bold green"
            case 1 | 2:
                color = "bold yellow"
            case _:
                color = "bold red"

    prediction_results = []

    for pred, log_likelihood, log_likelihood_target, n_incorrect in zip(preds, log_likelihoods, log_likelihoods_target, n_incorrects):
        for pred_key, pred_value in pred.items():
            if isinstance(pred_value, torch.Tensor):
                pred[pred_key] = pred_value.cpu().squeeze().numpy()
        result_per_sample = {
            **pred,
            "log_likelihood_post_hoc": log_likelihood,
            "log_likelihood_target": log_likelihood_target,
            "n_incorrect": n_incorrect,
        }
        prediction_results.append(result_per_sample)

    # save predictions with pickle
    save_path = os.path.join(PREDICTION_DIR, "predictions.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(prediction_results, f)
    # get size in MB
    size = os.path.getsize(save_path) / 1024 / 1024
    print(f"Saved {len(prediction_results)} predictions to {save_path}. Size: {size:.2f}MB")

    incorrect_counts = collections.Counter(n_incorrects)
    print("Incorrect Counts")
    for k, v in incorrect_counts.items():
        print(f"\t Error: {k} Count: {v}")
    print(f"Overall Accuracy: {incorrect_counts[0] / len(n_incorrects):.2f}")


if __name__ == "__main__":
    model_dir = os.path.join(cluster.artifact_dir, "models")

    main()

    print("Done!")
