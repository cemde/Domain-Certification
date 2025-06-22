import os
import time
from typing import Any, Dict, List, Tuple
from abc import ABC, abstractmethod

import rich
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.general import Timer
from utils import ddp

from .utilities import SequenceLengths, ScoringDataExample


SUBCATEGORIES = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}


MEDICAL_QA_SUB_CATEGORIES = set(
    [
        "anatomy",
        "clinical_knowledge",
        "college_medicine",
        "college_biology",
        "college_chemistry",
        # 'global_facts', # ??,
        "high_school_biology",
        "high_school_chemistry",
        "high_school_psychology",
        "human_aging",
        "human_sexuality",
        "medical_genetics",
        # 'miscellaneous', # ??,
        "nutrition",
        "professional_medicine",
        "virology",
    ]
)

# Categories that might have medical questions in them:
# Anthropology
# Human_Development_Index
# Hydrogen
# 'Animal'
# Gymnastics
# Sexual_orientation
# Bill_%26_Melinda_Gates_Foundation
# Poultry
# Child_labour
# Annelid Race_(human_categorization)
# Insect
# Endangered_Species_Act,
# Bird,
# Humanism,
# Oxygen

NOT_MEDICAL_QA_SUB_CATEGORIES = set(SUBCATEGORIES.keys()) - MEDICAL_QA_SUB_CATEGORIES


MMLU_SCORING_MARKER = [" A", " B", " C", " D"]


def mmlu_categories(family: str) -> List[str]:
    if family == "all":
        return list(SUBCATEGORIES.keys())
    elif family == "medical_qa":
        return list(MEDICAL_QA_SUB_CATEGORIES)
    elif family == "not_medical_qa":
        return list(NOT_MEDICAL_QA_SUB_CATEGORIES)
    else:
        raise ValueError(f"Invalid family {family}")


class MMLUDataset(Dataset, ABC):
    N_DEBUG: int = 256

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        root: str,
        N: int,
        split: str,
        debug: bool,
        n_shot: int,
        categories: List[str] = None,
        return_labels: bool = True,
        return_sequence: str = "full",
        return_answer: str = "correct",
        min_response_length: int = 0,
        truncate_sequence_to: int = None,
        use_chat_template: bool = False,
        training_task: str = "seq2seq",
    ):
        """MMLU dataset

        Args:
            tokenizer (AutoTokenizer): A Huggingface tokenizer from transformers library. Used to tokenize the data.
            root (str): The root directory where the data is stored. e.g. "/data/pubmedqa".
            N (int): Number of datapoints to subset the dataset to. If N<=0, then the entire dataset is used.
            split (str): The split of the dataset. Must be one of ["train", "val", "test"].
                Our "train" split is the official "auxiliary_train" split.
                Our "val" split is the official "val" split.
                Our "test" split is the official "test" split.
            debug (bool): If True, only a subset of the dataset is used for debugging purposes.
            n_shot (int): The number of n-shot examples to include in the dataset.
            categories (List[str], optional): The categories to include in the dataset. If None, all categories are included. Defaults to None.
            return_labels (bool, optional): If True, the labels are returned. Defaults to True.
            return_sequence (str, optional): The type of sequence to return. Must be one of ["full", "partial-random", "partial-response"]. Defaults to "full".
            return_answer (str, optional): Which answer choices to return. Must be one of ["correct", "random", "longest"]. Defaults to "correct".
            use_chat_template (bool, optional): If True, the tokenizer's chat_template is used to tokenize the data. Defaults to False.
            training_task (str, optional): The training task. Must be one of ["seq2seq", "lm"]. Defaults to "seq2seq".
            truncate_sequence_to (int, optional): The maximum length of the sequence. Defaults to None.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.N = N
        self.split = split
        self.debug = debug
        self.root = root
        self.dataset_num_token = None
        self.return_labels = return_labels
        self.return_sequence = return_sequence
        self.return_answer = return_answer
        self.min_response_length = min_response_length
        self.truncate_sequence_to = truncate_sequence_to
        self.use_chat_template = use_chat_template
        self.system_prompt = self.set_system_prompt()
        self.categories = categories
        self.training_task = training_task  # TODO this does not consistently do something. remove or fix.
        self.n_shot = n_shot

        self.validate_args()

        # Load from text file a list of all possibel SQuAD categories
        self.category2idx = {category: idx for idx, category in enumerate(SUBCATEGORIES.keys())}
        self.idx2category = {idx: category for idx, category in enumerate(SUBCATEGORIES.keys())}

        # Loading the official hf dataset
        dataset, self.n_shot_examples = self.load_dataset("cais/mmlu")
        self.dataset = self.process_data(dataset)

        # subset based on categories
        if categories is not None:
            self.dataset = self.dataset.filter(lambda x: x["subject"] in categories)

        # subset based on the argument min_response_length
        if self.min_response_length > 0:
            original_length = len(self.dataset)
            self.dataset = self.dataset.filter(
                lambda x: len(x["input_ids"]) - x["n_token_prompt"][0] > self.min_response_length
            )  # TODO this is the entire sequence. We need to filter on the response length.
            ddp.print(
                f"Filtered out {original_length - len(self.dataset):,} samples with response length < {min_response_length}. Remaining: {len(self.dataset):,}"
            )
            assert len(self.dataset) > 0, "No samples left after filtering for min_response_length"

        # subset based on the argument N
        if self.N > 0:
            self.dataset = self.dataset.select(range(self.N))
            ddp.pprint(f"[bold red]WARNING:  Subsetting the MMLU dataset (split={self.split}) to {self.N} examples[/bold red]")

        print(f"Loaded {len(self.dataset):,} examples from {self.__class__.__name__} dataset.")

        # keep track of sequence lengths
        self.seq_lengths = SequenceLengths()
        self.seq_lengths.extend([x.as_py() for x in list(self.dataset.data["length"])])
        ddp.print(
            f"Min sequence length: {self.seq_lengths.min()} | Mean sequence length: {self.seq_lengths.mean():.2f} | Max sequence length: {self.seq_lengths.max()}"
        )

        # initialize time to track dataloading time
        self.timer = Timer()

    def validate_args(self):
        """Validate the arguments passed to the dataset"""
        assert self.return_sequence in ["partial-response", "partial-random", "full"], (
            f"Invalid return_sequence. Got {self.return_sequence}. Must be one of ['full', 'partial-response', 'partial-random']"
        )
        assert self.split in ["train", "val", "test"], f"Invalid split. Got {self.split}. Must be one of ['train', 'val', 'test']"
        assert not self.use_chat_template or self.tokenizer.chat_template is not None, "Tokenizer does not have chat_template"
        assert all([c in SUBCATEGORIES for c in self.categories]), f"Invalid categories. Got {self.categories}. Must be a subset of categories."
        assert self.truncate_sequence_to is None or self.truncate_sequence_to > 0, (
            f"Invalid truncate_sequence_to. Got {self.truncate_sequence_to}. Must be > 0"
        )
        assert self.min_response_length >= 0, f"Invalid min_response_length. Got {self.min_response_length}. Must be >= 0"
        assert self.return_answer in ["correct", "random", "longest"], (
            f"Invalid return_answer. Got {self.return_answer}. Must be one of ['correct', 'random', 'longest']"
        )
        if not self.return_answer == "correct":
            ddp.print(f"[bold yellow]WARNING: return_answer={self.return_answer} is not 'correct'. Only use for training.")
        if self.debug:
            rich.print(f"[bold red]WARNING: DEBUG Mode on for dataset {self.__class__.__name__}[/bold red]")

    def set_system_prompt(self) -> str:
        # TODO rework if chat template is used
        return "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."

    def load_dataset(self, dataset_name_or_path) -> Tuple[Dataset, Dataset]:
        # decide on dataset type
        builder = "all"
        split = {"train": "auxhiliary_train", "val": "validation", "test": "test"}[self.split]
        # load the raw dataset from HF and load "dev" dataet containing n-shot examples
        if self.split in ["val", "test"]:
            dataset = load_dataset(dataset_name_or_path, builder, split=split, trust_remote_code=False, cache_dir=self.root)
            n_shot_examples = load_dataset(dataset_name_or_path, "all", split="dev", trust_remote_code=False, cache_dir=self.root)
        else:
            split = "train"  # this datasets calls it "train" instead of "auxiliary_train"
            dataset = load_dataset(
                "kz919/mmlu-auxiliary-train-auto-labelled", "default", split=split, trust_remote_code=False, cache_dir=self.root
            )
            n_shot_examples = load_dataset(dataset_name_or_path, "all", split="dev", trust_remote_code=False, cache_dir=self.root)
            # rename "task" to "subject" to match the other datasets
            dataset = dataset.remove_columns("subject")
            dataset = dataset.rename_column("task", "subject")
        # subset if debug mode is enalbed
        if self.debug:
            dataset = dataset.select(range(min(self.N_DEBUG, len(dataset))))
        return dataset, n_shot_examples

    @abstractmethod
    def get_cache_file_name(self) -> str:
        pass

    def process_data(self, dataset):
        # Process and tokenize the dataset
        os.makedirs(self.root + "/caches", exist_ok=True)
        cache_file_name = self.get_cache_file_name()

        return dataset.map(
            self.process_and_tokenize_function,
            remove_columns=dataset.column_names,
            cache_file_name=cache_file_name,
            num_proc=16,
        )

    @abstractmethod
    def process_and_tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def __len__(self) -> int:
        return len(self.dataset)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pass

    @staticmethod
    def get_dataset_config_name(cfg) -> str:
        return f"{cfg.data.name}_{cfg.data.categories}_{cfg.data.n_shot}"


class MMLUQADataset(MMLUDataset):
    """This dataset class inteprets the MMLU dataset as a question answering dataset.

    The samples are formatted as follows:
    Question: <question>
    Context: <context>
    Answer: <correct answer>

    Example:::
    Question: What is the capital of France?
    Context: France is a country in Europe.

    """

    def get_cache_file_name(self) -> str:
        return f"{self.root}/caches/mmlu_qa_{self.split}_{f'debug{self.N_DEBUG}_' if self.debug else ''}{'_'.join(self.tokenizer.name_or_path.split('/')).replace('@', '_')}_{'chatTemplate' if self.use_chat_template else 'noChatTemplate'}_maxLength{self.truncate_sequence_to}.arrow"

    def validate_args(self) -> None:
        super().validate_args()
        assert self.n_shot == 0, "N-shot not implemented yet for this dataset in QA mode."
        # if not self.return_answer == "correct" and not self.split == "train":
        #     raise ValueError("Only 'correct' answers are available for the validation and test splits.")

    def format_qa(self, question: str) -> str:
        if question[-1] == "?":
            return f"Question: {question}\n\nAnswer: "
        else:
            return f"{question} "

    def process_and_tokenize_function(self, example):
        question = example["question"]
        correct_answer_id = example["answer"]
        if self.return_answer == "correct":
            answer = example["choices"][correct_answer_id]
        elif self.return_answer == "random":
            answer = example["choices"][torch.randint(0, 4, (1,)).item()]
        elif self.return_answer == "longest":
            answer = max(example["choices"], key=len)
        else:
            raise ValueError(f"Invalid return_answer. Got {self.return_answer}. Must be one of ['correct', 'random', 'longest']")

        if self.use_chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Question: {question}"},
                {"role": "assistant", "content": answer},
            ]
            tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", truncation=True)
            prompt_tokenized = self.tokenizer.apply_chat_template(messages[:-1], return_tensors="pt", add_generation_prompt=True)
            raise NotImplementedError("Chat template not implemented/validated yet")
        else:
            # input_text = f"Question: {question}"
            # transition_text = "\n\nAnswer: "
            prompt_text = self.format_qa(question)

            # Get the length of the prompt
            input_and_transition_tokens = self.tokenizer(prompt_text)
            input_and_transition_length = len(input_and_transition_tokens["input_ids"])

            # Tokenize the combined text
            combined_text = prompt_text + answer
            tokenized = self.tokenizer(combined_text, return_tensors="pt")
            # Warning: due to the BPE tokenization, this is not exact. it can sometimes be one token more or one token to few.

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # Add EOS token if not already present
        if input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids = torch.cat([input_ids, torch.tensor([self.tokenizer.eos_token_id])])
            attention_mask = torch.cat([attention_mask, torch.tensor([1])])

        labels = input_ids.clone()
        if self.training_task == "seq2seq":
            labels[:input_and_transition_length] = -100

        # assert sequence charactereistics
        assert input_ids.shape == attention_mask.shape, f"input_ids.shape={input_ids.shape} != attention_mask.shape={attention_mask.shape}"
        assert input_ids[0] == self.tokenizer.bos_token_id, f"input_ids[0]={input_ids[0]} != bos_token_id={self.tokenizer.bos_token_id}"
        assert input_ids[-1] == self.tokenizer.eos_token_id, f"input_ids[-1]={input_ids[-1]} != eos_token_id={self.tokenizer.eos_token_id}"

        # due to the possible imprecision of the tokenization, we truncate the sequence to truncate_sequence_to in the response:
        if self.truncate_sequence_to is not None:
            input_ids = input_ids[: self.truncate_sequence_to]
            attention_mask = attention_mask[: self.truncate_sequence_to]
            labels = labels[: self.truncate_sequence_to]
            assert input_ids.shape[0] <= self.truncate_sequence_to, (
                f"input_ids.shape[0]={input_ids.shape[0]} > truncate_sequence_to={self.truncate_sequence_to}"
            )

        return {
            **example,
            "subject_id": torch.tensor([self.category2idx[example["subject"]]]),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "n_token_prompt": torch.tensor([input_and_transition_length]),
            "len_prompt_text": torch.tensor([len(prompt_text)]),
            "length": len(input_ids),
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        start_time = time.time()
        example = self.dataset[idx]

        return_dict = {
            "input_ids": example["input_ids"],
            # "attention_mask": example["attention_mask"],
            "n_token_prompt": example["n_token_prompt"],
            "len_prompt_text": example["len_prompt_text"],
            "idx": torch.tensor([idx]),
            "subject_id": example["subject_id"],
        }

        if self.return_sequence == "partial-random":
            split_point = torch.randint(0, len(example["input_ids"]), (1,)).item()
            return_dict["input_ids"] = [self.tokenizer.bos_token_id] + example["input_ids"][split_point:]
            # return_dict["attention_mask"] = example["attention_mask"][:split_point]
            return_dict["n_token_prompt"] = torch.tensor([1])

        elif self.return_sequence == "partial-response":
            split_point = example["n_token_prompt"][0]
            return_dict["input_ids"] = [self.tokenizer.bos_token_id] + example["input_ids"][split_point:]
            # return_dict["attention_mask"] = example["attention_mask"][:split_point]
            return_dict["n_token_prompt"] = torch.tensor([1])

        if self.return_labels:
            return_dict["labels"] = example["labels"]

        for k, v in return_dict.items():
            if isinstance(v, list):
                return_dict[k] = torch.tensor(v)

        self.timer.append(time.time() - start_time)

        return return_dict


class MMLUScoringDataset(MMLUDataset):
    """This dataset class inteprets the MMLU dataset as a question answering dataset.

    The samples are formatted as follows:
    Header

    [N-shot examples]

    <Question>
    A: <choice a>
    B: <choice b>
    C: <choice c>
    D: <choice d>
    Answer:

    Example (zero-shot):::
    Header

    What is the capital of France?
    A: The capital of France is Paris.
    B: The capital of France is Berlin.
    C: The capital of France is London.
    D: The capital of France is Madrid.
    Answer:

    """

    def get_cache_file_name(self):
        return f"{self.root}/caches/mmlu_scoring_{self.split}_{f'debug{self.N_DEBUG}_' if self.debug else ''}{'_'.join(self.tokenizer.name_or_path.split('/')).replace('@', '_')}_{'chatTemplate' if self.use_chat_template else 'noChatTemplate'}_{self.n_shot}_maxLength{self.truncate_sequence_to}.arrow"

    def validate_args(self):
        super().validate_args()
        assert self.n_shot in [0, 1, 5], f"Invalid n_shot. Got {self.n_shot}. Must be one of [0, 5]"
        assert self.truncate_sequence_to is None, (
            f"sequence truncation not implemented yet for {self.__class__.__name__}. Must be None. Got truncate_sequence_to={self.truncate_sequence_to}"
        )
        assert self.min_response_length == 0, (
            f"All sequence lengths must be included for {self.__class__.__name__}. Got min_response_length={self.min_response_length}"
        )

    @staticmethod
    def format_multiple_choice(example: Dict[str, Any], add_correct_answer: bool = False) -> Dict[str, Any]:
        question = example["question"]
        choice_a, choice_b, choice_c, choice_d = example["choices"]
        correct_answer_id = example["answer"]

        text = f"{question}\nA. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}\nAnswer:"

        if add_correct_answer:
            correct_answer_marker = ["A", "B", "C", "D"][correct_answer_id]
            text += f" {correct_answer_marker}"

        return text

    def process_and_tokenize_function(self, example):
        # n-shot examples
        n_shot_examples = self.n_shot_examples.filter(lambda x: x["subject"] == example["subject"])
        n_shot_examples = n_shot_examples.select(range(self.n_shot))
        n_shot_examples_texts = [self.format_multiple_choice(example, add_correct_answer=True) for example in n_shot_examples]

        # question and answer
        subject = example["subject"].replace("_", " ")
        instruction_header = f"The following are multiple choice questions (with answers) about {subject}."
        item = self.format_multiple_choice(example, add_correct_answer=False)
        correct_answer_id = example["answer"]

        if self.use_chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Question: {question}"},
                {"role": "assistant", "content": answer},
            ]
            tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", truncation=True)
            prompt_tokenized = self.tokenizer.apply_chat_template(messages[:-1], return_tensors="pt", add_generation_prompt=True)
            raise NotImplementedError("Chat template not implemented/validated yet")
        else:
            all_examples = "\n\n".join(n_shot_examples_texts) if self.n_shot > 0 else ""
            entire_text = "{instruction_header}\n\n{n_shot_examples}\n\n{item}".format(
                instruction_header=instruction_header, n_shot_examples=all_examples, item=item
            )
            tokenized = self.tokenizer(entire_text, return_tensors="pt")

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        # Add EOS token if not already present
        if input_ids[-1] == self.tokenizer.eos_token_id:
            input_ids = input_ids[:-1]
            attention_mask = attention_mask[:-1]

        labels = input_ids.clone()

        return {
            **example,
            "n_shot_examples": [x for x in n_shot_examples],
            "subject_id": torch.tensor([self.category2idx[example["subject"]]]),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "n_token_prompt": torch.tensor([len(input_ids)]),
            "len_prompt_text": torch.tensor([len(entire_text)]),
            "length": len(input_ids),
            "correct_answer_id": torch.tensor([correct_answer_id]),
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor | List[str]]:
        start_time = time.time()
        example = self.dataset[idx]
        return_dict = {"idx": torch.tensor([idx]), **example}
        del return_dict["answer"]  # # remove from example: answer

        # convert Lists[NumericDataTypes] to Tensors
        for k, v in return_dict.items():
            if k == "n_shot_examples":
                continue
            if isinstance(v, list) and all([isinstance(x, (int, float)) for x in v]):
                return_dict[k] = torch.tensor(v)

        self.timer.append(time.time() - start_time)

        # this makes sure that all datasets output the same format.
        return_object = ScoringDataExample.from_dict(return_dict)
        # for compatibility with the old code, we return a dictionary
        return return_object.to_dict()
