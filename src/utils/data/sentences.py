from typing import Dict, Union
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class SentenceDataset(Dataset):
    def __init__(self, rootdir: str, truncate_text_to: int, N: int, tokenizer: AutoTokenizer, split: str = "test", return_string: bool = False):
        self.rootdir = rootdir
        self.split = split
        self.file_path = f"{rootdir}/{split}.txt"
        self.truncate_text_to = truncate_text_to
        self.N = N
        self.tokenizer = tokenizer
        self.return_string = return_string

        # Load text data
        with open(self.file_path, "r") as file:
            self.data = file.readlines()

        self.data = [text.strip() for text in self.data]

        # truncate
        self.data = [text[: self.truncate_text_to] for text in self.data]
        self.data = self.data[: self.N]

        # Tokenize
        if not self.return_string:
            self.tokenized_data = [self.tokenizer(text, return_tensors="pt") for text in self.data]
            for i in range(len(self.tokenized_data)):
                self.tokenized_data[i]["n_token_prompt"] = torch.tensor([1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Union[str, torch.Tensor]]:
        if self.return_string:
            data = {"input_text": self.data[idx]}
        else:
            data = self.tokenized_data[idx]
        return data | {"idx": idx}

    @staticmethod
    def get_dataset_config_name(cfg):
        return f"{cfg.data.topic}"


# Example usage:
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# dataset = TextDataset(rootdir='data', max_length=128, tokenizer=tokenizer)
# print(dataset[0])
