# ruff: noqa: E402
import os
import pickle
import copy
import sys
from typing import Any, Dict, List, Tuple

import rich
from tqdm import tqdm
import utils.cluster
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, DataCollatorForLanguageModeling

from utils.data import (
    get_dataset_config_name,
    TaskDatasetTraining,
    PubMedQADataset,
    SQuADDataset,
    PubMedQAWithGeneratedResponsesDataset,
    SQuADWithGeneratedResponsesDataset,
    MMLUQADataset,
    mmlu_categories,
)
from utils.collators import DataCollatorForLanguageModelingTimed, DataCollatorForSeq2SeqTimed
import utils
from utils.process_logits import postprocess_logits
from utils.inference import token_log_likelihood

torch.set_printoptions(linewidth=220)


class DataCollatorExtendingInt:
    def __init__(self, data_collator):
        self.data_collator = data_collator

    def __call__(self, examples, *args, **kwargs):
        for i, example in enumerate(examples):
            for k, v in example.items():
                if isinstance(v, int):
                    examples[i][k] = torch.tensor([v])
        return self.data_collator(examples, *args, **kwargs)


def get_data(cfg, tokenizer: AutoTokenizer, model: AutoModelForCausalLM):
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
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
        }

        dataset = TaskDatasetTraining(N=cfg.data.test_size, split="test", **kwargs)

    elif cfg.data.name == "pubmedqa":
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "PubMedQA"),
            "debug": cfg.data.debug,
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "truncate_sequence_to": cfg.data.truncate_sequence_to,
            "use_chat_template": cfg.data.use_chat_template,
        }
        dataset = PubMedQADataset(N=cfg.data.test_size, split="test", **kwargs)

    elif cfg.data.name == "pubmedqa_generated":
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "PubMedQAWithGeneratedResponses"),
            "debug": cfg.data.debug,
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "truncate_sequence_to": cfg.data.truncate_sequence_to,
            "use_chat_template": cfg.data.use_chat_template,
            "generated_output_path": os.path.join(cluster.artifact_dir, cfg.data.generated_output_paths.test),
            "keep_judge_decision": cfg.data.keep_judge_decision,
        }
        dataset = PubMedQAWithGeneratedResponsesDataset(N=cfg.data.test_size, split="test", **kwargs)

    elif cfg.data.name == "squad":
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "SQuAD"),
            "debug": cfg.data.debug,
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "truncate_sequence_to": cfg.data.truncate_sequence_to,
            "use_chat_template": cfg.data.use_chat_template,
            "categories": cfg.data.categories,
        }
        dataset = SQuADDataset(N=cfg.data.test_size, split="test", **kwargs)

    elif cfg.data.name == "squad_generated":
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "SQuADWithGeneratedResponses"),
            "debug": cfg.data.debug,
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "truncate_sequence_to": cfg.data.truncate_sequence_to,
            "min_response_length": cfg.data.min_response_length,
            "sequence_prefix": None if cfg.data.sequence_prefix in ["None", "NONE", "none", ""] else cfg.data.sequence_prefix,
            "use_chat_template": cfg.data.use_chat_template,
            "categories": cfg.data.categories,
            "generated_output_path": os.path.join(cluster.artifact_dir, cfg.data.generated_output_path),
            "keep_judge_decision": None
            if cfg.data.keep_judge_decision in [None, "None", "NONE", "none", "all", "ALL", "All"]
            else cfg.data.keep_judge_decision,
        }
        dataset = SQuADWithGeneratedResponsesDataset(N=cfg.data.test_size, split="test", **kwargs)

    elif cfg.data.name == "mmlu":
        assert cfg.data.variant in ["qa"]
        truncate_to = None if cfg.data.truncate_sequence_to in ["None", "NONE", "none", ""] else int(cfg.data.truncate_sequence_to)
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "MMLU"),
            "debug": cfg.data.debug,
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "truncate_sequence_to": truncate_to,
            "use_chat_template": cfg.data.use_chat_template,
            "categories": mmlu_categories(cfg.data.categories),
            "n_shot": cfg.data.n_shot,
        }
        dataset = MMLUQADataset(N=cfg.data.test_size, split="test", **kwargs)

    else:
        raise ValueError(f"Unknown dataset {cfg.data.name}")

    if cfg.inference.task == "seq2seq":
        data_collator = DataCollatorForSeq2SeqTimed(tokenizer, model=model, padding=True)
    elif cfg.inference.task == "causal":
        data_collator = DataCollatorForLanguageModelingTimed(tokenizer, mlm=False)
    else:
        raise ValueError(f"Unknown task {cfg.training.task}")

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.inference.batch_size,
        collate_fn=lambda x: x,
        drop_last=cfg.inference.drop_last_batch,
        shuffle=cfg.inference.shuffle_batches,
    )

    # test data collator
    try:
        b = [dataset[i] for i in range(8)]
        b = data_collator(b)
    except Exception as e:
        print(f"DataCollator Error: {e}")

    # print some examples
    return dataset, dataloader, data_collator


def get_models(cfg: DictConfig):
    # get model
    if cfg.model.source == "hf":
        model_name_or_path = cfg.model.name_or_path
    elif cfg.model.source == "local":
        model_name_or_path = os.path.join(model_dir, cfg.model.name_or_path)
    try:
        kwargs = {"config": AutoConfig.from_pretrained(model_name_or_path)}
    except OSError:
        print(f"Could not load Model AutoConfig from {model_name_or_path}.")
        kwargs = {}
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, cache_dir=model_dir, device_map=cfg.model.device, **kwargs)
    model = utils.models.set_model_precision(model, cfg.model.precision)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"Model={cfg.model.name_or_path}. Size: {model_size:,} ({model_size / (1000**2):.2f}M) parameters.")

    # get generator
    if cfg.generator.source == "hf":
        generator_name_or_path = cfg.generator.name_or_path
    elif cfg.generator.source == "local":
        generator_name_or_path = os.path.join(model_dir, cfg.generator.name_or_path)
    try:
        kwargs = {"config": AutoConfig.from_pretrained(generator_name_or_path)}
    except OSError:
        print(f"Could not load Model AutoConfig from {generator_name_or_path}.")
        kwargs = {}
    generator = AutoModelForCausalLM.from_pretrained(generator_name_or_path, cache_dir=model_dir, device_map=cfg.generator.device, **kwargs)
    generator = utils.models.set_model_precision(generator, cfg.generator.precision)
    generator_size = sum(t.numel() for t in generator.parameters())
    print(f"Model={cfg.generator.name_or_path}. Size: {generator_size:,} ({generator_size / (1000**2):.2f}M) parameters.")

    return model, generator


def move_to_cpu(outputs):
    keys = vars(outputs).keys()
    for k in keys:
        if isinstance(getattr(outputs, k), torch.Tensor):
            setattr(outputs, k, getattr(outputs, k).cpu())
    return outputs


def get_token_prompt_length(example: Dict[str, torch.Tensor], prompt_length: str) -> int:
    if prompt_length == "dataset":
        return example["n_token_prompt"][0]
    elif prompt_length == "random":
        raise NotImplementedError("Random prompt length not implemented.")
    else:
        return int(prompt_length)


def truncate_inputs(
    tokenizer: AutoTokenizer, batch: Dict[str, torch.Tensor], distribution: str, prompt_length: str
) -> Dict[str, torch.Tensor | List[int]]:
    assert distribution in ["y|x", "x+y", "y"]
    if not distribution == "y":
        # add "prompt_length" to batch
        for i, example in enumerate(batch):
            n_token_prompt = get_token_prompt_length(example, prompt_length)
            batch[i]["prompt_length"] = [n_token_prompt]
        return batch

    new_batch = []
    for i, example in enumerate(batch):
        input_length = len(example["input_ids"])
        n_token_prompt = get_token_prompt_length(example, prompt_length)
        new_example = {}
        for k, v in example.items():
            if isinstance(v, torch.Tensor) and len(v) == input_length:
                # truncate sequences.
                if k in ["input_ids", "labels"]:
                    # if sequence of tokens prepend with bos token
                    new_example_sequence = v[n_token_prompt - 1 :]
                    new_example_sequence[0] = tokenizer.bos_token_id
                    new_example[k] = new_example_sequence
                else:
                    new_example[k] = v[n_token_prompt - 1 :]
            else:
                new_example[k] = v
        new_example["prompt_length"] = [n_token_prompt]
        new_batch.append(new_example)
    return new_batch


def truncate_logits_and_sequence(
    tokenizer: AutoTokenizer, batch: Dict[str, torch.Tensor], logits: torch.Tensor, distribution: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the sequence and logits for evaluation.

    Assume sequence <s> 5 3 A 3 5 </s> with prompt <s> 5 3 A:
        * If distribution == "y|x" this returns 3 5 </s> with according logits.
        * If distribution == "y" this returns 3 5 </s> with according logits.
        * If distribution == "x+y" this returns <s> 5 3 A 3 5 </s> with according logits.

    Note: logits[i] predict for token[i].

    Args:
        tokenizer (AutoTokenizer): _description_
        batch (Dict[str, torch.Tensor]): _description_
        logits (torch.Tensor): _description_
        distribution (str): _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    assert logits.shape[:2] == batch["input_ids"].shape  # check shapes are compatable

    # if y|x, all the x must be removed. if y or x+y, only the BOS needs to be handled.
    if not distribution == "y|x":
        # the first token is BOS.
        # logits inform the next token:
        #    logits[i] for token i+1. logits[i] = log p(x_{i+1}|x_{0:i})
        #    logits[i] => token_id [i+1]
        return logits[:, :-1, :], batch["input_ids"][:, 1:]

    # get number of padding tokens per sequence
    n_zeros_per_row = batch["input_ids"].eq(tokenizer.pad_token_id).sum(-1)

    # generate new, truncated sequences of tokens and logits
    new_logits = []
    seqs = []
    max_length = 0
    for logits_seq, n_token, inp_seq, n_pad in zip(logits, batch["prompt_length"], batch["input_ids"], n_zeros_per_row):
        # truncate token sequence
        seqs.append(inp_seq[n_token:-n_pad] if n_pad > 0 else inp_seq[n_token:])
        # truncate logits. shifted to left, because logits[i] => token[i+1]
        logits_seq = logits_seq[n_token - 1 : -n_pad - 1] if n_pad > 0 else logits_seq[n_token - 1 : -1]
        new_logits.append(logits_seq)
        max_length = max(max_length, len(logits_seq))

    # pad everything to max length
    for i, (seq, l_seq) in enumerate(zip(seqs, new_logits)):
        if len(l_seq) < max_length:
            pad_length = max_length - len(l_seq)
            padding = torch.full((pad_length, l_seq.shape[-1]), tokenizer.pad_token_id, device=l_seq.device)
            new_logits[i] = torch.cat([l_seq, padding])
            padding = torch.full((pad_length,), tokenizer.pad_token_id, device=seq.device)
            seqs[i] = torch.cat([seq, padding])
        assert seqs[i].shape[0] == new_logits[i].shape[0]

    return torch.stack(new_logits), torch.stack(seqs)


def dict_to_device(d, device):
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.to(device)
    return d


def subset_dict(d, keys):
    return {k: v for k, v in d.items() if k in keys}


def get_token_prompt_length_from_text(text: str, len_prompt_text: int, tokenizer: AutoTokenizer) -> int:
    """Returns the number of tokens in the prompt based on the text and the length of the prompt text.

    Args:
        text (str): The text to be tokenized.
        len_prompt_text (int): The length of the prompt in the text.
        tokenizer (AutoTokenizer): The tokenizer to be used.

    Returns:
        int: The number of tokens in the prompt.
    """

    seq = tokenizer.encode(text[:len_prompt_text])
    if tokenizer.eos_token_id is not None and seq[-1] == tokenizer.eos_token_id:
        seq = seq[:-1]
    return len(seq)


def get_generator_batch(
    cfg: DictConfig,
    model_batch: Dict[str, torch.Tensor],
    tokenizer_model: AutoTokenizer,
    tokenizer_generator: AutoTokenizer,
) -> Dict[str, torch.Tensor]:
    # if the tokenizers are the same, return the model batch.
    if cfg.tokenizers_match:
        return model_batch

    generator_batch = []
    for example in model_batch:
        # translate
        text = tokenizer_model.decode(example["input_ids"].tolist(), skip_special_tokens=True)
        # get the prompt length for the generator
        n_token_prompt_generator = torch.tensor([get_token_prompt_length_from_text(text, example["len_prompt_text"], tokenizer_generator)])

        # encode the text using the generator tokenizer
        example_generator = tokenizer_generator.encode(text, return_tensors="pt").squeeze()
        if not example_generator[-1] == tokenizer_generator.eos_token_id:
            example_generator = torch.cat([example_generator, torch.tensor([tokenizer_generator.eos_token_id])])

        # add the prompt length, idx, len_prompt_text to the example
        generator_batch.append(
            {
                "input_ids": example_generator,
                "idx": example["idx"],
                "len_prompt_text": example["len_prompt_text"],
                "n_token_prompt": n_token_prompt_generator,
            }
        )

    # check that the keys are the same
    assert set(generator_batch[0].keys()) == set(model_batch[0].keys())
    # check that the dimensions of each value are the same (only the first element in batch is checked)
    assert all([model_batch[0][k].ndim == generator_batch[0][k].ndim for k in model_batch[0].keys()])
    # check dtypes are the same
    assert all([model_batch[0][v].dtype == generator_batch[0][v].dtype for v in model_batch[0].keys()])
    # check that len_prompt_text and idx are the same
    assert all([model_batch[0][v] == generator_batch[0][v] for v in ["len_prompt_text", "idx"]])

    return generator_batch


def inference(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    generator: AutoModelForCausalLM,
    tokenizer_model: AutoTokenizer,
    tokenizer_generator: AutoTokenizer,
    dataloader: DataLoader,
    data_collator: DataCollatorForLanguageModeling,
) -> Tuple[List[Any], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Returns the batch, token sequences and the logits for model and generator.

    Assume a sequence "<s> 5 3 A R E S 6 4 </s>" and prompting "<s> 5 3 A R E S":
        * If distribution == "y|x" the output will be for "6 4 </s>".
        * If distribution == "y" the output will be for "6 4 </s>".
        * If distribution == "x+y" the output will be for "<s> 5 3 A R E S 6 4 </s>".

    Note: The return sequences and logits do not contain the BOS token, <s>.

    Args:
        cfg (DictConfig): _description_
        model (AutoModelForCausalLM): _description_
        generator (AutoModelForCausalLM): _description_
        tokenizer (AutoTokenizer): _description_
        dataloader (DataLoader): _description_
        data_collator (DataCollatorForLanguageModeling): _description_

    Returns:
        Tuple[List[Any], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]: batch, sequence, logits for model, logits for generator
    """
    logits_model_list = []
    logits_generator_list = []
    batch_model_list = []
    batch_generator_list = []
    seq_model_list = []
    seq_generator_list = []

    model.eval()
    generator.eval()

    for i, batch_model in enumerate(tqdm(dataloader, desc="Inference")):
        # create batch for generator and save both batches
        batch_generator = get_generator_batch(cfg, batch_model, tokenizer_model, tokenizer_generator)
        batch_model_list.append(batch_model)
        batch_generator_list.append(batch_generator)

        # truncate inputs depending on target distribution
        batch_model = truncate_inputs(tokenizer_model, copy.deepcopy(batch_model), cfg.model.target_distribution, cfg.inference.prompt_length)
        batch_generator = truncate_inputs(
            tokenizer_generator, copy.deepcopy(batch_generator), cfg.generator.target_distribution, cfg.inference.prompt_length
        )

        # collate batch
        batch_model = data_collator(batch_model)
        batch_generator = data_collator(batch_generator)

        assert "labels" in batch_model, "Labels not found in batch for model."
        assert "labels" in batch_generator, "Labels not found in batch for generator."

        # move to device
        batch_model = dict_to_device(batch_model, model.device)
        batch_generator = dict_to_device(batch_generator, generator.device)

        keys_forward_pass = ["input_ids", "labels"]

        with torch.inference_mode():
            outputs_model = model(**subset_dict(batch_model, keys_forward_pass))  # seq: BOS + X + Y + EOS
            outputs_generator = generator(**subset_dict(batch_generator, keys_forward_pass))  # seq: BOS + Y + EOS

            logits_model = outputs_model.logits.cpu()
            logits_generator = outputs_generator.logits.cpu()

            # truncate logits if needed depending on target distribution
            logits_model, seq_model = truncate_logits_and_sequence(tokenizer_model, batch_model, logits_model, cfg.model.target_distribution)
            logits_generator, seq_generator = truncate_logits_and_sequence(
                tokenizer_generator, batch_generator, logits_generator, cfg.generator.target_distribution
            )

            # ensure all tensors are on the same device: cpu
            logits_model, logits_generator = logits_model.cpu(), logits_generator.cpu()
            seq_model, seq_generator = seq_model.cpu(), seq_generator.cpu()

            # assert shapes match
            if cfg.tokenizers_match:
                assert logits_model.shape == logits_generator.shape
                assert (seq_model == seq_generator).all()

            assert seq_model.shape == logits_model.shape[:2], (
                f"Shape mismatch for model return data: {seq_model.shape} != {logits_model.shape[:2]}"
            )
            assert seq_generator.shape == logits_generator.shape[:2], (
                f"Shape mismatch for generator return data: {seq_generator.shape} != {logits_generator.shape[:2]}"
            )

            seq_model_list.append(seq_model)
            seq_generator_list.append(seq_generator)
            logits_model_list.append(logits_model)  # logits[i] predicts token[i]
            logits_generator_list.append(logits_generator)  # logits[i] predicts token[i]

    return batch_model_list, batch_generator_list, seq_model_list, seq_generator_list, logits_model_list, logits_generator_list


def token_entropy(
    tokenizer: AutoTokenizer,
    token_ids: torch.Tensor,
    logsoftmax: torch.Tensor,
    mask_padding: bool = True,
    mask_eos: bool = True,
) -> torch.Tensor:
    # get entropy of a sentence for all tokens.

    assert token_ids.dim() == 2, f"sentence should be 2D, got {token_ids.dim()}. batch x tokens"
    assert logsoftmax.dim() == 3, f"logsoftmax should be 3D, got {logsoftmax.dim()}. batch x tokens x vocab"
    assert token_ids.shape[0] == logsoftmax.shape[0], "batch size mismatch"
    assert token_ids.shape[1] == logsoftmax.shape[1], "sequence length mismatch"

    # Get the entropy of the response_ids
    entropy = -torch.sum(logsoftmax * torch.exp(logsoftmax), dim=-1)

    # mask out padding tokens
    if mask_padding:
        mask = token_ids.ne(tokenizer.pad_token_id)
        entropy = entropy * mask

    # remove EOS token
    if mask_eos:
        mask = mask & token_ids.ne(tokenizer.eos_token_id)
        entropy = entropy * mask

    return entropy


def split_x_y(cfg, tokenizer, batch):
    """Split the input into x and y."""
    x = []
    y = []
    for example in batch:
        n_token_prompt = get_token_prompt_length(example, cfg.inference.prompt_length)
        x_text = example["input_ids"][:n_token_prompt]
        y_text = example["input_ids"][n_token_prompt:]
        x.append(tokenizer.decode(x_text, skip_special_tokens=True))
        y.append(tokenizer.decode(y_text, skip_special_tokens=True))
    return x, y


def assert_decoder_vocab_size(tokenizer: AutoTokenizer, seq: torch.Tensor, logits: torch.Tensor) -> None:
    target_shapes = [(*seq.shape, tokenizer.vocab_size)]
    try:
        target_shapes.append((*seq.shape, tokenizer.vocab_size + len(tokenizer.added_tokens_decoder)))
    except AttributeError:
        pass
    assert logits.shape in target_shapes, f"Logits shape {logits.shape} does not match target shapes {target_shapes}."


def calculate_log_likelihoods(
    cfg,
    tokenizer_model,
    tokenizer_generator,
    batches_model,
    batches_generator,
    sequences_model,
    sequences_generator,
    outputs_model,
    outputs_generator,
):
    """Expects sequences without BOS."""
    len_prompt_text_list = []
    n_token_prompt_model_list, n_token_prompt_generator_list = [], []
    input_ids_model_list, input_ids_generator_list = [], []
    log_likelihoods_model_list, log_likelihoods_generator_list = [], []
    entropy_model_list, entropy_generator_list = [], []
    sequence_length_model_list, sequence_length_generator_list = [], []
    x_model_list, y_model_list = [], []
    x_generator_list, y_generator_list = [], []

    assert (
        sum([len(b) for b in batches_model])
        == sum([len(b) for b in batches_generator])
        == sum([len(b) for b in sequences_model])
        == sum([len(b) for b in sequences_generator])
        == sum([len(b) for b in outputs_model])
        == sum([len(b) for b in outputs_generator])
    )

    for batch_model, batch_generator, seq_model, seq_generator, logits_model, logits_generator in tqdm(
        zip(batches_model, batches_generator, sequences_model, sequences_generator, outputs_model, outputs_generator),
        desc="Calculating Log Likelihoods",
        total=len(batches_model),
    ):
        # some tokenizers have added decoder tokens that are not in the vocab. Thse need to be added to match the logits shape.
        assert_decoder_vocab_size(tokenizer_model, seq_model, logits_model)
        assert_decoder_vocab_size(tokenizer_generator, seq_generator, logits_generator)

        if cfg.tokenizers_match:
            assert logits_model.shape == logits_generator.shape, (
                f"Logits shapes do not match: {logits_model.shape} != {logits_generator.shape}. Required when tokenizers match."
            )

        # postprocess logits
        logits_model = postprocess_logits(logits_model, cfg.model.temperature, cfg.model.top_k)
        logits_generator = postprocess_logits(logits_generator, cfg.generator.temperature, cfg.generator.top_k)

        # calculate log likelihood
        log_likelihoods_model, logsoftmax_model = token_log_likelihood(tokenizer_model, seq_model, logits_model)
        log_likelihoods_generator, logsoftmax_generator = token_log_likelihood(tokenizer_generator, seq_generator, logits_generator)

        # calculate entropy per token
        entropy_model = token_entropy(tokenizer_model, seq_model, logsoftmax_model)
        entropy_generator = token_entropy(tokenizer_generator, seq_generator, logsoftmax_generator)

        # split text into x and y
        x_model, y_model = split_x_y(cfg, tokenizer_model, batch_model)
        x_generator, y_generator = split_x_y(cfg, tokenizer_generator, batch_generator)

        # append to list
        n_token_prompt_model_list.append(torch.tensor([x["n_token_prompt"] for x in batch_model]))
        n_token_prompt_generator_list.append(torch.tensor([x["n_token_prompt"] for x in batch_generator]))
        len_prompt_text_list.append(torch.tensor([x["len_prompt_text"] for x in batch_model]))
        input_ids_model_list.append(seq_model)
        input_ids_generator_list.append(seq_generator)
        log_likelihoods_model_list.append(log_likelihoods_model)
        log_likelihoods_generator_list.append(log_likelihoods_generator)
        entropy_model_list.append(entropy_model)
        entropy_generator_list.append(entropy_generator)
        x_model_list.extend(x_model)
        y_model_list.extend(y_model)
        x_generator_list.extend(x_generator)
        y_generator_list.extend(y_generator)
        sequence_length_model_list.extend([len(b["input_ids"]) for b in batch_model])
        sequence_length_generator_list.extend([len(b["input_ids"]) for b in batch_generator])

    # concatenate
    n_token_prompt_model_list = torch.cat(n_token_prompt_model_list, dim=0)
    n_token_prompt_generator_list = torch.cat(n_token_prompt_generator_list, dim=0)
    len_prompt_text_list = torch.cat(len_prompt_text_list, dim=0)
    input_ids_model_list = torch.cat(utils.batched_right_padding(input_ids_model_list, 1, 0), dim=0)
    input_ids_generator_list = torch.cat(utils.batched_right_padding(input_ids_generator_list, 1, 0), dim=0)
    log_likelihoods_model_list = torch.cat(utils.batched_right_padding(log_likelihoods_model_list, 1, 0), dim=0)
    log_likelihoods_generator_list = torch.cat(utils.batched_right_padding(log_likelihoods_generator_list, 1, 0), dim=0)
    entropy_model_list = torch.cat(utils.batched_right_padding(entropy_model_list, 1, 0), dim=0)
    entropy_generator_list = torch.cat(utils.batched_right_padding(entropy_generator_list, 1, 0), dim=0)

    return {
        "n_token_prompt_model": n_token_prompt_model_list.numpy(),
        "n_token_prompt_generator": n_token_prompt_generator_list.numpy(),
        "len_prompt_text": len_prompt_text_list.numpy(),
        "input_ids_model": input_ids_model_list.numpy(),
        "input_ids_generator": input_ids_generator_list.numpy(),
        "log_likelihoods_model": log_likelihoods_model_list.numpy(),
        "log_likelihoods_generator": log_likelihoods_generator_list.numpy(),
        "entropy_model": entropy_model_list.numpy(),
        "entropy_generator": entropy_generator_list.numpy(),
        "x_text_model": x_model_list,
        "y_text_model": y_model_list,
        "x_text_generator": x_generator_list,
        "y_text_generator": y_generator_list,
        "sequence_length_model": sequence_length_model_list,
        "sequence_length_generator": sequence_length_generator_list,
    }


def validate_cfg(cfg):
    if isinstance(cfg.inference.prompt_length, int):
        cfg.inference.prompt_length = str(cfg.inference.prompt_length)
    assert utils.verify_prompt_length_argument(cfg.inference.prompt_length), "Invalid prompt length argument."

    if cfg.inference.prompt_length == "dataset":
        assert "seq2seq" in cfg.data.training_tasks, (
            f"Prompt length is set to dataset, but dataset {cfg.data.name} does not support seq2seq training or inference."
        )

    if not cfg.tokenizers_match:
        assert cfg.inference.prompt_length == "dataset", "Prompt length must be dataset when using different tokenizers."


@hydra.main(config_path="config", config_name="model_likelihood", version_base="1.3")
def main(cfg: DictConfig):
    # create experiment name
    cfg.data_config_name = get_dataset_config_name(cfg)
    EXPERIMENT_NAME = utils.generate_id()
    RESULT_DIR = os.path.join(
        cluster.artifact_dir,
        "model_likelihood",
        EXPERIMENT_NAME,
    )

    validate_cfg(cfg)

    # save config
    os.makedirs(RESULT_DIR, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(RESULT_DIR, "config.yaml"))

    # if the script is called from another script using subprocess the stdout might be broken.
    printing_manager = utils.PrintingManager(
        os.path.join(cluster.log_dir, "model_likelihood", f"output_log_{EXPERIMENT_NAME}.txt") if cfg.log.print_to_file else None
    )

    # print experiment name and directory to stdout and stderr
    rich.print(f"[bold green]EXPERIMENT NAME: {EXPERIMENT_NAME} [/bold green]")
    rich.print(f"[bold green]Using Directory: {RESULT_DIR} for predictions.[/bold green]")
    print(f"EXPERIMENT NAME: {EXPERIMENT_NAME}", file=sys.stderr)
    print(f"Using Directory: {RESULT_DIR} for predictions.", file=sys.stderr)

    # set seed
    utils.seed_everything(cfg.run.seed)

    # get tokenizers
    tokenizer_model, tokenizer_generator = utils.tokenizers.get_tokenizers(cfg, model_dir)

    # get model
    model, generator = get_models(cfg)
    if cfg.model.tokenizer.add_pad_token:
        model.resize_token_embeddings(len(tokenizer_model))
    if cfg.generator.tokenizer.add_pad_token:
        generator.resize_token_embeddings(len(tokenizer_generator))

    # data
    dataset, dataloader, data_collator = get_data(cfg, tokenizer_model, model)

    # forward pass through model
    batches_model, batches_generator, sequences_model, sequences_generator, logits_model, logits_generator = inference(
        cfg, model, generator, tokenizer_model, tokenizer_generator, dataloader, data_collator
    )

    # compute the log likelihoods
    results = calculate_log_likelihoods(
        cfg,
        tokenizer_model,
        tokenizer_generator,
        batches_model,
        batches_generator,
        sequences_model,
        sequences_generator,
        logits_model,
        logits_generator,
    )

    # save predictions with pickle
    save_path = os.path.join(RESULT_DIR, "model_likelihood.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"config": cfg, "data": results}, f)
    size = os.path.getsize(save_path) / 1024 / 1024  # get size in MB
    print(f"Saved {len(results)} predictions to {save_path}. Size: {size:.2f}MB")

    printing_manager.reset()

    return {"config": cfg, "data": results, "save_path": save_path, "experiment_name": EXPERIMENT_NAME}


if __name__ == "__main__":
    cluster = utils.cluster.ClusterManager()
    model_dir = os.path.join(cluster.artifact_dir, "models")

    main()
