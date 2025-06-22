import os
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TinyShakespeareDataset(Dataset):
    def __init__(
        self,
        N: int,
        tokenizer: AutoTokenizer,
        context_length: int,
        root: str,
        return_sequence: str,
        prepend_bos: bool = False,
        append_eos: bool = False,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.context_length = context_length
        self.root = root
        self.return_sequence = return_sequence
        self.N = N
        self.name = "TinyShakespeare"
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos

        assert self.return_sequence in ["full", "prompt"]
        assert self.split in ["train", "val", "test"]
        if not self.return_sequence == "full":
            raise NotImplementedError("Only full sequence generation is supported.")

        with open(os.path.join(self.root, f"{split}.txt"), "r") as f:
            dataset = [f.read()]

        tokenized_datasets = self.tokenize_function(dataset, self.tokenizer)
        tokenized_datasets["input_ids"][0] = tokenized_datasets["input_ids"][0][1:]  # remove "global" BOS token
        self.data = self.convert_data_to_blocks(tokenized_datasets)
        N_original = len(self.data["input_ids"])
        if self.N > 0:
            self.data = {k: self.data[k][:N] for k in self.data.keys()}
            print(f"TinyShakespeare: {self.split} split. Loaded {N_original} examples from the dataset, truncated to {N} examples.")
        else:
            print(f"TinyShakespeare: {self.split} split. Loaded {N_original} examples from the dataset.")

    def tokenize_function(self, examples, tokenizer):
        return tokenizer(examples, return_special_tokens_mask=True)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {
            "input_ids": torch.tensor(self.data["input_ids"][idx]),
            "n_token_prompt": torch.tensor([-1]),
            "len_prompt_text": torch.tensor([-1]),
        }

    def __len__(self):
        return len(self.data["input_ids"])

    def convert_data_to_blocks(self, examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        seq_length = self.context_length - int(self.prepend_bos) - int(self.append_eos)
        if total_length >= seq_length:
            total_length = (total_length // seq_length) * seq_length
        result = {k: [t[i : i + seq_length] for i in range(0, total_length, seq_length)] for k, t in concatenated.items()}
        for i, seq in enumerate(result["input_ids"]):
            if self.prepend_bos:
                seq = [self.tokenizer.bos_token_id] + seq
            if self.append_eos:
                seq = seq + [self.tokenizer.eos_token_id]
            result["input_ids"][i] = seq
        return result
