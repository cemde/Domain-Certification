import json
import os
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class DatasetFromFile(Dataset):
    """Purpose of this dataset class is not to be used for a specific established dataset, but rather a loose collection of prompts and reponse pairs, that"""

    def __init__(
        self,
        N: int,
        tokenizer: AutoTokenizer,
        root: str,
        fileformat: str,
        return_sequence: str,
        debug: bool = False,
        prepend_bos: bool = False,
        append_eos: bool = False,
        split: str = "train",
        truncate_sequence_to: int = -1,
    ):
        self.tokenizer = tokenizer
        self.split = split
        self.root = root
        self.name = "DebugData"
        self.return_sequence = return_sequence
        self.N = N
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.truncate_sequence_to = truncate_sequence_to
        self.fileformat = fileformat

        assert self.return_sequence in ["full", "prompt"]
        assert self.split in ["train", "val", "test"]
        if not self.return_sequence == "full":
            raise NotImplementedError("Only full sequence generation is supported.")

        dataset = self._load_data()

        tokenized_data = []

        for example in dataset["data"]:
            tokenized_example = {
                "prompt": self.tokenizer.encode(example["prompt"], add_special_tokens=True),
                "prompt_text": example["prompt"],
                "response": self.tokenizer.encode(example["response"], add_special_tokens=True),
                "response_text": example["response"],
            }
            tokenized_data.append(tokenized_example)

        self.data = tokenized_data

        N_original = len(self.data)
        if self.N > 0:
            self.data = {k: self.data[k][:N] for k in self.data.keys()}
            print(f"DatasetFromFile: {self.split} split. Loaded {N_original} examples from the dataset, truncated to {N} examples.")
        else:
            print(f"DatasetFromFile: {self.split} split. Loaded {N_original} examples from the dataset.")

    def _load_data(self):
        if self.fileformat == "txt":
            assert False, "Not implemented"
            with open(os.path.join(self.root, f"{self.split}.txt"), "r") as f:
                dataset = [f.read()]
        elif self.fileformat == "json":
            with open(os.path.join(self.root, f"{self.split}.json"), "r") as f:
                dataset = json.load(f)
        else:
            raise NotImplementedError(f"File format {self.fileformat} is not supported.")
        return dataset

    def tokenize_function(self, examples, tokenizer):
        return tokenizer(examples, return_special_tokens_mask=True)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        example = self.data[idx]
        prompt = example["prompt"]
        response = example["response"][1:]  # remove the first token, which is the bos token
        prompt_text = example["prompt_text"]

        if self.append_eos and response[-1] != self.tokenizer.eos_token_id:
            response = response + [self.tokenizer.eos_token_id]

        if self.prepend_bos and prompt[0] != self.tokenizer.bos_token_id:
            prompt = [self.tokenizer.bos_token_id] + prompt

        return {
            "input_ids": torch.tensor(prompt + response),
            "n_token_prompt": torch.tensor([len(prompt)]),
            "len_prompt_text": torch.tensor([len(prompt_text)]),
            "idx": torch.tensor([idx]),
        }

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_dataset_config_name(cfg) -> str:
        return f"DatasetFromFile_{cfg.data.directory}_test"
