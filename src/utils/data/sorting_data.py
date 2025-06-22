import itertools
import json
import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import rich
import torch
from pydantic import BaseModel
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.general import Timer

TASKS = ["sort", "reverse-sort", "add", "even-odd"]

NUMBERS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
LETTERSNUMBERS = "abcdefghijklmnopqrstuvwxyz0123456789"

all_combinations = [" ".join(x) for x in list(itertools.permutations(["A", "E", "R", "S"]))]
QUERY_TOKEN = "Q "

# Randomly smples answer token
ANSWER_TOKENS = {
    "sort": [f" {x} " for x in all_combinations if x.startswith("S")],
    "reverse-sort": [f" {x} " for x in all_combinations if x.startswith("R")],
    "add": [f" {x} " for x in all_combinations if x.startswith("A")],
    "even-odd": [f" {x} " for x in all_combinations if x.startswith("E")],
}

# # Deterministic answer tokens. rotations of A E R S
# ANSWER_TOKENS = {
#     "sort": [" S A E R "],
#     "reverse-sort": [" R S A E "],
#     "add": [" A E R S "],
#     "even-odd": [" E R S A "],
# }


class TupleIndexIterator:
    def __init__(self, n: int, length: int):
        self.n = n
        self.current = [0]
        self.idx = 0
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):
        # check max length
        if self.idx >= self.length:
            raise StopIteration
        self.idx += 1

        current = tuple(self.current)

        self.current[-1] += 1
        for i in range(len(self.current) - 1, -1, -1):
            if self.current[i] >= self.n:
                self.current[i] = 0
                if i == 0:
                    self.current.insert(0, 0)
                else:
                    self.current[i - 1] += 1
        return current


def get_composite_population(num_int: int, num_char) -> torch.Tensor:
    int_population = np.arange(1, num_int + 1).astype(str)
    char_population = get_single_population(num_char, LETTERS)
    return np.concatenate((int_population, char_population))


def get_single_population(length: int, pool: str) -> torch.Tensor:
    population = []
    index_iterator = TupleIndexIterator(len(pool), length)
    for indices in index_iterator:
        word = "".join([pool[j] for j in indices])
        population.append(word)

    return np.array(population)


def generate_sequence(population: np.ndarray, length: int, max_length: int) -> np.ndarray:
    sample_indices = np.random.permutation(max_length)[:length]
    return population[sample_indices]


def perform_task(permuted_subseq: np.ndarray, population: np.ndarray, task: str) -> np.ndarray:
    if task == "sort":
        return np.sort(permuted_subseq)
    elif task == "reverse-sort":
        return np.sort(permuted_subseq)[::-1]
    elif task == "add":
        population_shifted = {x: y for x, y in zip(population, np.roll(population, -1))}
        return np.array([population_shifted[x] for x in permuted_subseq], dtype=permuted_subseq.dtype)
    elif task == "even-odd":
        order = [sum([ord(char) for char in word]) for word in permuted_subseq]
        order = [x % 2 for x in order]
        permuted_subseq = [x for _, x in sorted(zip(order, permuted_subseq))]
        return np.array(permuted_subseq)
    else:
        raise ValueError("Invalid task")


def get_population(modality: str, num_int: int, num_char: int) -> Tuple[np.ndarray, int]:
    if modality == "int":
        population = np.arange(1, num_int + 1).astype(str)
        L = num_int
    elif modality == "char":
        population = get_single_population(num_char, LETTERS)
        L = num_char
    elif modality == "mixed":
        population = get_composite_population(num_int, num_char)  # get_single_population(L, LETTERSNUMBERS)
        L = num_int + num_char
    else:
        raise ValueError("Invalid modality")
    return population, L


def generate_sequences(
    modality: str, task: str, N: int, T: int, num_int: int, num_char: int, constant_length: bool = True
) -> List[Tuple[str, str]]:
    population, L = get_population(modality, num_int, num_char)

    sequences = []
    # Step 1: Generate a sequence of integers from 1 to L (inclusive)
    for _ in tqdm(range(N), desc="Generating sequences"):
        # Step 2: Randomly select T integers from the sequence to form a subsequence
        max_length = T if constant_length else torch.randint(1, T + 1, (1,)).item()
        permuted_subseq = generate_sequence(population, max_length, L)

        # Step 4: Sort the subsequence in ascending order
        response_subseq = perform_task(permuted_subseq, population, task)

        # convert to comma separated strings
        permuted_subseq = " ".join(map(lambda x: str(x.item()), permuted_subseq))
        response_subseq = " ".join(map(lambda x: str(x.item()), response_subseq))

        # Collect the permuted and sorted subsequences
        sequences.append((permuted_subseq, response_subseq))

    return sequences


class TaskDatasetConfig(BaseModel):
    seed: int
    modality: str
    task: str
    n_train: int
    n_val: int
    n_test: int
    split: str
    n: int
    max_seq_length: int
    num_int: int
    num_char: int
    constant_seq_length: bool


class TaskDataset(Dataset):
    Config = TaskDatasetConfig

    def __init__(
        self,
        modality: str,
        task: str,
        N: int,
        max_seq_length: int,
        num_int: int,
        num_char: int,
        split: str,
        constant_seq_length: bool,
        tokenizer: AutoTokenizer,
        root: str,
        return_labels: bool = True,
        return_sequence: str = "full",
    ):
        self.modality = modality
        self.task = task
        self.N = N
        self.T = max_seq_length
        self.num_int = num_int
        self.num_char = num_char
        self.split = split
        self.constant_seq_length = constant_seq_length
        self.tokenizer = tokenizer
        self.root = root
        self.return_labels = return_labels
        self.return_sequence = return_sequence

        # parital sequence only during training. Check which Mixin.
        assert return_sequence in ["full", "partial-random", "partial-response"]
        if not isinstance(self, TrainingGetter) and not self.return_sequence == "full":
            raise ValueError(f"Partial sequence only available during training. This is dataset class: {self.__class__.__name__}")

        self.timer = Timer()

        self._possible_tokens = {"q": QUERY_TOKEN, "a": ANSWER_TOKENS[task]}
        self.tokens = {"q": lambda: QUERY_TOKEN, "a": lambda: random.sample(ANSWER_TOKENS[task], 1)[0]}

        self.population = get_population(modality, num_int, num_char)

        load_path = os.path.join(
            root, self.get_disk_subpath(modality, task, max_seq_length, num_int, num_char, constant_seq_length), f"{split}.json"
        )
        try:
            with open(load_path, "r") as f:
                loaded_data = json.load(f)
                self.sequences = loaded_data["data"]
                self.loaded_config = TaskDatasetConfig(**loaded_data["config"])
            print(f"Loaded {len(self.sequences):,} sequences from {load_path}.")
        except FileNotFoundError:
            raise FileNotFoundError(f"'{self.__class__.__name__}' can't load data. File {load_path} with data not found.")
        if N < len(self.sequences):
            print(f"Truncating to {N=:,} sequences.")
        elif N > len(self.sequences):
            raise ValueError(f"Requested {N=:,} sequences, but only {len(self.sequences):,} sequences are available.")

        self.sequences = self.sequences[:N]

    @staticmethod
    def get_disk_subpath(
        dataset_modality: str, task: str, max_sequence_length: int, num_int: int, num_char: int, constant_sequence_length: bool
    ):
        return f"MOD_{dataset_modality}_TASK_{task}_LEN_{max_sequence_length}_INT_{num_int}_CHAR_{num_char}_CSL_{constant_sequence_length}"

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx) -> Tuple[str, str]:
        raise NotImplementedError("This method is not implemented.")

    def get_longest_sequence(self) -> int:
        length = 0
        for seq in (pbar := tqdm(self, desc="Getting longest sequence")):
            length = max(length, len(seq["input_ids"]))
            pbar.set_postfix({"LEN": length})
        return length

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__} (Modality: {self.modality}, Task: {self.task})"

    @staticmethod
    def get_dataset_config_name(cfg) -> str:
        return f"{cfg.data.name}_{cfg.data.modality}_{cfg.data.task}"

    # def check_correctness(self, sequence: torch.Tensor) -> Tuple[float, float]:
    #     input_sequence, predicted_sequence = split(sequence)
    #     target = perform_task(input_sequence, self.population, self.task)
    #     accuracy = (target == predicted_sequence).float().mean().item()
    #     num_mismatches = ((target - predicted_sequence).abs() > 0).float().mean().item()
    #     return accuracy, num_mismatches


class TrainingGetter:
    def __getitem__(self, idx) -> Tuple[str]:
        start_time = time.time()
        permuted, sorted_seq = self.sequences[idx]
        query_token = self.tokens["q"]()
        q_permuted = permuted
        answer_token = self.tokens["a"]()
        a_sorted_seq = sorted_seq
        eos_token = ""
        # prompt: e.g. Q 1 5 3 S E A R
        prompt = query_token + q_permuted + answer_token
        response = a_sorted_seq + eos_token

        # randomly cut off the sequence. keep the second part.
        if self.return_sequence == "partial-random":
            split_point = random.randint(0, len(prompt) + len(response) - 1)
            # keep past the split point
            if split_point < len(prompt):
                prompt = prompt[split_point:]
                response = response
            else:
                original_prompt_length = len(prompt)
                prompt = ""
                response = response[split_point - original_prompt_length :]

        if self.return_sequence == "partial-response":
            prompt = ""
            response = response

        combined = prompt + response
        len_prompt_text = len(prompt)
        prompt_ids = self.tokenizer.encode(prompt)[:-1]  # remove eos_token
        len_prompt_ids = len(prompt_ids)

        return_dict = self.tokenizer(combined, return_tensors="pt")

        if self.return_labels:
            labels = return_dict["input_ids"].clone()
            labels[:, :len_prompt_ids] = -100
            return_dict["labels"] = labels

        return_dict = {k: v.squeeze() for k, v in return_dict.items()}
        return_dict["n_token_prompt"] = [len_prompt_ids]
        return_dict["len_prompt_text"] = [len_prompt_text]
        self.timer.append(time.time() - start_time)
        # input_ids: Entire Sequence
        # labels [optional]: Entire Sequence with -100 for inputs
        # n_token_prompt: Number of tokens in the prompt
        # len_prompt_text: Length of the prompt text
        return return_dict


class PromptingGetter:
    """returns a dictionary with input_ids (prompt), target_ids (entire sequence)."""

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        permuted, sorted_seq = self.sequences[idx]
        input_text = self.tokens["q"]() + permuted + self.tokens["a"]()
        target_text = input_text + sorted_seq  # + self.tokenizer.eos_token
        input_ids = self.tokenizer.encode(input_text)
        target_ids = self.tokenizer.encode(target_text)
        # test_input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        # test_target_text = self.tokenizer.decode(target_ids, skip_special_tokens=False)
        return {"input_ids": torch.tensor(input_ids), "target_ids": torch.tensor(target_ids), "n_token_prompt": len(input_ids)}


class TaskCollectionDataset(Dataset):
    def __init__(self, N: int, datasets: List[TaskDataset], shuffle_tasks: bool = True):
        self.datasets = datasets
        assert all([datasets[0].split == dataset.split for dataset in datasets])
        self.T = max([dataset.T for dataset in datasets])
        self.split = datasets[0].split
        # self.tokenizer = tokenizer
        # self._possible_tokens = {"q": QUERY_TOKEN, "a": ANSWER_TOKENS[task]}
        # self.tokens = {"q": lambda: QUERY_TOKEN, "a": lambda: random.sample(ANSWER_TOKENS[task], 1)[0]}

        self.timer = Timer()

        # redirect idx to the correct dataset and sample
        self.dataset_idx = []
        for idx in range(len(self.datasets)):
            self.dataset_idx.extend([idx] * len(datasets[idx]))

        self.sample_idx = []
        for idx in range(len(self.datasets)):
            self.sample_idx.extend(list(range(len(datasets[idx]))))

        # shuffle tasks
        if shuffle_tasks:
            random.shuffle(self.dataset_idx)

        # shuffle samples within tasks
        datasets_N = sum([len(dataset) for dataset in datasets])
        if N < datasets_N:
            print(f"Truncating to {N=:,} sequences.")
            if not shuffle_tasks:
                rich.print("[yellow]Warning: Truncating without shuffling tasks might eliminate some tasks.[/yellow]")
        elif N > datasets_N:
            raise ValueError(f"Requested {N=:,} sequences, but only {datasets_N:,} sequences are available.")

        self.dataset_idx = self.dataset_idx[:N]
        self.sample_idx = self.sample_idx[:N]

        print(f"Collection Dataset of length {len(self):,} created. Split={self.split}. Contains {len(datasets)} datasets:")
        for dataset in self.datasets:
            print(f"\t- {len(dataset):,} sequences from {dataset.__class__.__name__} ({dataset.modality}, {dataset.task})")

    def __len__(self):
        return len(self.dataset_idx)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        dataset_idx = self.dataset_idx[index]
        return self.datasets[dataset_idx][self.sample_idx[index]]

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}"


class TaskDatasetTraining(TrainingGetter, TaskDataset):
    pass


class TaskDatasetPrompting(PromptingGetter, TaskDataset):
    pass


def print_examples(dataset: TaskDataset, split: str, tokenizer: AutoTokenizer, num_examples: int = 2):
    print(f"Examples from {split} set:")
    samples = random.sample(range(len(dataset)), num_examples)
    for i in samples:
        seq = dataset[i]["input_ids"]
        seq = tokenizer.decode(seq)
        print(f"\tInput: {seq}")
        try:
            seq = dataset[i]["labels"]
            seq = seq[seq >= 0]
            seq = tokenizer.decode(seq)
            print(f"\tLabel: {seq}")
        except KeyError:
            print("\tLabel: <not in batch>")
