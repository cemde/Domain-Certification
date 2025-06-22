from typing import Any, Dict, List
import re

import numpy as np
import rich
import torch

from ..data.sorting_data import get_population, perform_task


def is_valid_instruction_sequence(seq):
    # Regular expression to find potential sequences
    pattern = r"\b([AERS])\s([AERS])\s([AERS])\s([AERS])\b"
    matches = re.findall(pattern, seq)

    if not len(matches) == 1:
        return False

    # Check if any sequence has unique and correct letters
    for match in matches:
        if len(set(match)) == 4:  # Ensures all characters are unique
            return True
    return False


def find_instruction_index(seq_list):
    index = 0
    for i, token in enumerate(seq_list):
        if token in ["A", "E", "R", "S"]:
            index = i
            break
    return index


def _score_task_data(pred: torch.Tensor, modality: str, num_int: int, num_char: int, verbose: bool) -> Dict[str, Any]:
    seq = pred["prompt_prediction_text"]
    seq_list = np.array(seq.split(" "))

    # starts with Q
    starts_with_q = seq_list[0] == "Q"

    # sequence has instruction tokens
    instruct_sequence_valid = is_valid_instruction_sequence(seq)

    if verbose:
        print(pred["prompt_prediction_text"], end=" ")
        print(f"{starts_with_q=} {instruct_sequence_valid=}", end=" ")

    if not (starts_with_q and instruct_sequence_valid):
        return {
            "starts_with_q": False,
            "instruct_sequence_valid": False,
            "lengths_match": None,
            "task_correct": False,
            "int_population": None,
            "num_token_match": None,
        }

    # split into parts
    Q_seq_list = seq_list[0]

    # find index instruction sequence in remaining sequence
    idx_instruct = find_instruction_index(seq_list)

    permuted_seq = seq_list[1:idx_instruct]
    instruction_seq = seq_list[idx_instruct : idx_instruct + 4]
    completed_seq = seq_list[idx_instruct + 4 :]

    lengths_match = len(permuted_seq) == len(completed_seq)

    if verbose:
        print(f"{lengths_match=}")

    # check whether task is fullfilled
    task_name = {
        "A": "add",
        "E": "even-odd",
        "R": "reverse-sort",
        "S": "sort",
    }[instruction_seq[0]]

    population, L = get_population(modality, num_int, num_char)
    expected_completed_seq = perform_task(permuted_seq, population, task_name)
    expected_full_sequence = [" ".join(x) for x in [[Q_seq_list], permuted_seq, instruction_seq, expected_completed_seq]]
    expected_full_sequence = " ".join(expected_full_sequence)

    # number of matching elements seq vs expected_full_sequence
    array_expected_seq = np.array(expected_completed_seq)
    array_completed_seq = np.array(completed_seq)
    min_length = min(len(array_expected_seq), len(array_completed_seq))
    num_token_match = (array_expected_seq[:min_length] == array_completed_seq[:min_length]).mean()

    if lengths_match:
        task_correct = (completed_seq == expected_completed_seq).all()
        if verbose:
            col = "green" if task_correct else "red"
            rich.print(f"[{col}]{expected_full_sequence}[/{col}] {num_token_match=:.2f}")
    else:
        task_correct = False
        if verbose:
            rich.print(
                f"[red]{expected_full_sequence}[/red] len(permuted)={len(permuted_seq)} len(completed)={len(completed_seq)} {num_token_match=:.2f}"
            )

    # modality is mixed
    int_population = set(get_population("int", num_int, num_char)[0])
    permuted_seq_is_int = set(permuted_seq).issubset(int_population)

    if verbose:
        print()

    return {
        "starts_with_q": starts_with_q,
        "instruct_sequence_valid": instruct_sequence_valid,
        "lengths_match": lengths_match,
        "task_correct": task_correct,
        "int_population": permuted_seq_is_int,
        "num_token_match": num_token_match,
    }


def _aggregate_scores_task_data(preds: List[Dict[str, int]]) -> Dict[str, float]:
    # aggregate results

    int_population = []
    task_correct = []
    instruct_sequence_valid = []
    lengths_match = []
    sequences_text = []
    num_token_match = []

    for pred in preds:
        int_population.append(pred["results"]["int_population"])
        task_correct.append(pred["results"]["task_correct"])
        instruct_sequence_valid.append(pred["results"]["instruct_sequence_valid"])
        lengths_match.append(pred["results"]["lengths_match"])
        sequences_text.append(pred["prompt_prediction_text"])
        num_token_match.append(pred["results"]["num_token_match"])

    int_population = np.array(int_population).astype(float)
    task_correct = np.array(task_correct).astype(float)
    instruct_sequence_valid = np.array(instruct_sequence_valid).astype(float)
    lengths_match = np.array(lengths_match).astype(float)
    sequences_text = set(sequences_text)
    num_token_match = np.array(num_token_match).astype(float)

    task_correct_int = task_correct[int_population.astype(bool)]

    # score None as false.
    # mean = lambda x: np.nanmean(x)
    def mean(x):
        return np.mean(np.nan_to_num(x, nan=0))

    return {
        "int_population": mean(int_population),
        "task_correct": mean(task_correct),
        "instruct_sequence_valid": mean(instruct_sequence_valid),
        "lengths_match": mean(lengths_match),
        "unique_sequences": len(sequences_text) / len(preds),
        "num_token_match": mean(num_token_match),
        "task_correct_int_population": mean(task_correct_int),
    }
