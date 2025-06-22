from dataclasses import dataclass
import torch


class SequenceLengths:
    def __init__(self):
        self.seqs = []

    def __getitem__(self, idx):
        return self.seqs[idx]

    def append(self, x):
        self.seqs.append(x)

    def mean(self):
        return sum(self.seqs) / len(self.seqs)

    def min(self):
        return min(self.seqs)

    def max(self):
        return max(self.seqs)

    def extend(self, x):
        self.seqs.extend(x)


# thanks DeepSeek R1


@dataclass
class ScoringDataExample:
    idx: torch.Tensor
    question: str
    subject: str
    choices: list[str]
    n_shot_examples: list[dict]
    subject_id: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    n_token_prompt: torch.Tensor
    len_prompt_text: torch.Tensor
    length: int
    correct_answer_id: torch.Tensor

    def __post_init__(self):
        # Type assertions
        assert isinstance(self.question, str), "question must be str"
        assert isinstance(self.subject, str), "subject must be str"
        assert isinstance(self.choices, list) and all(isinstance(c, str) for c in self.choices), "choices must be list of str"
        assert isinstance(self.n_shot_examples, list), "n_shot_examples must be list"
        assert isinstance(self.length, int), "length must be int"

        # Tensor type assertions
        tensor_fields = ["idx", "subject_id", "input_ids", "attention_mask", "labels", "n_token_prompt", "len_prompt_text", "correct_answer_id"]
        for field in tensor_fields:
            tensor = getattr(self, field)
            assert isinstance(tensor, torch.Tensor), f"{field} must be torch.Tensor"
            assert tensor.dtype in [torch.int32, torch.int64, torch.long], f"{field} must be integer tensor"

        # Dimension assertions
        assert self.idx.dim() == 1 and len(self.idx) == 1, "idx must be 1D tensor of length 1"
        assert self.subject_id.dim() == 1, "subject_id must be 1D tensor"
        assert self.input_ids.dim() == 1, "input_ids must be 1D tensor"
        assert self.attention_mask.dim() == 1, "attention_mask must be 1D tensor"
        assert self.labels.dim() == 1, "labels must be 1D tensor"
        assert self.n_token_prompt.dim() == 1, "n_token_prompt must be 1D tensor"
        assert self.len_prompt_text.dim() == 1, "len_prompt_text must be 1D tensor"
        assert self.correct_answer_id.dim() == 1, "correct_answer_id must be 1D tensor"

        # Matching lengths
        assert len(self.input_ids) == len(self.attention_mask) == len(self.labels), (
            "input_ids, attention_mask, and labels must have same length"
        )
        assert len(self.input_ids) == self.length, "length must match input_ids length"

        # Choices validation
        assert len(self.choices) >= 2, "must have at least 2 choices"

    def to_dict(self) -> dict:
        return {
            "idx": self.idx,
            "question": self.question,
            "subject": self.subject,
            "choices": self.choices,
            "n_shot_examples": self.n_shot_examples,
            "subject_id": self.subject_id,
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
            "n_token_prompt": self.n_token_prompt,
            "len_prompt_text": self.len_prompt_text,
            "length": self.length,
            "correct_answer_id": self.correct_answer_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ScoringDataExample":
        # Convert lists to tensors
        tensor_fields = ["idx", "subject_id", "input_ids", "attention_mask", "labels", "n_token_prompt", "len_prompt_text", "correct_answer_id"]
        for field in tensor_fields:
            if isinstance(data[field], list):
                data[field] = torch.tensor(data[field], dtype=torch.long)

        return cls(**data)
