import hashlib
import json
import os
import time
from typing import Any, Dict, List

import rich
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.general import Timer
from utils import ddp

from .utilities import SequenceLengths

MEDICAL_QA_CATEGORIES = set(
    [
        "Antibiotics",
        "Symbiosis",
        "Gene",
        "Brain",
        "Immunology",
        "Biodiversity",
        "Digestion",
        "Pharmaceutical_industry",
        "Mammal",
        "Nutrition",
        "Tuberculosis",
        "On_the_Origin_of_Species",
        "Asthma",
        "Pain",
        "Bacteria",
        "Infection",
        "Black_Death",
        "Pharmacy",
        "Immune_system",
        "Chloroplast",
    ]
)  # Anthropology, Human_Development_Index, Hydrogen, 'Animal' Gymnastics Sexual_orientation Bill_%26_Melinda_Gates_Foundation Poultry Child_labour Annelid Race_(human_categorization) Insect Endangered_Species_Act, Bird, Humanism, Oxygen


class SQuADDataset(Dataset):
    N_DEBUG: int = 256

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        root: str,
        N: int,
        split: str,
        debug: bool,
        categories: str = "all",
        return_labels: bool = True,
        return_sequence: str = "full",
        min_response_length: int = 0,
        sequence_prefix: str = None,
        truncate_sequence_to: int = None,
        use_chat_template: bool = False,
        training_task: str = "seq2seq",
    ):
        """SQuAD dataset

        Args:
            tokenizer (AutoTokenizer): A Huggingface tokenizer from transformers library. Used to tokenize the data.
            root (str): The root directory where the data is stored. e.g. "/data/pubmedqa".
            N (int): Number of datapoints to subset the dataset to. If N<=0, then the entire dataset is used.
            split (str): The split of the dataset. Must be one of ["train", "val", "test"].
            debug (bool): If True, only a subset of the dataset is used for debugging purposes.
            return_sequence (str, optional): The type of sequence to return. Must be one of ["full", "partial-random", "partial-response"]. Defaults to "full".
            min_response_length (int, optional): The minimum length of the response in tokens. Shorter responses are filtered out. Defaults to 0.
            sequence_prefix (str, optional): The prefix to add to the sequence. Defaults to None.
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
        self.min_response_length = min_response_length
        self.sequence_prefix = sequence_prefix
        self.truncate_sequence_to = truncate_sequence_to
        self.use_chat_template = use_chat_template
        self.system_prompt = self.set_system_prompt()
        self.training_task = training_task
        self.categories = categories

        self.validate_args()

        # Load from text file a list of all possibel SQuAD categories
        self.idx2category = self.load_categories()
        self.category2idx = {c: i for i, c in enumerate(self.idx2category)}

        # Loading the official hf dataset
        dataset = self.load_dataset("rajpurkar/squad")
        self.dataset = self.process_data(dataset)

        # subset based on categories
        if categories == "medical_qa":
            self.dataset = self.dataset.filter(lambda x: x["title"] in MEDICAL_QA_CATEGORIES)
        elif categories == "not_medical_qa":
            self.dataset = self.dataset.filter(lambda x: x["title"] not in MEDICAL_QA_CATEGORIES)

        # capture the sequence lengths
        self.seq_lengths = SequenceLengths()
        self.seq_lengths.extend([x.as_py() for x in list(self.dataset.data["length"])])
        ddp.print(
            f"Min sequence length: {self.seq_lengths.min()} | Mean sequence length: {self.seq_lengths.mean():.2f} | Max sequence length: {self.seq_lengths.max()}"
        )

        # capture the reponse lengthts and filter out samples with response length < min_response_length
        self.response_lengths = SequenceLengths()
        self.response_lengths.extend([x.as_py() for x in list(self.dataset.data["n_token_response"])])
        if min_response_length > 0:
            original_length = len(self.dataset)
            self.dataset = self.dataset.filter(lambda x: x["n_token_response"] > min_response_length)
            ddp.print(
                f"Filtered out {original_length - len(self.dataset):,} samples with response length < {min_response_length}. Remaining: {len(self.dataset):,}"
            )
            assert len(self.dataset) > 0, "No samples left after filtering for min_response_length"
            ddp.print(
                f"Min sequence length: {self.seq_lengths.min()} | Mean sequence length: {self.seq_lengths.mean():.2f} | Max sequence length: {self.seq_lengths.max()}"
            )

        # subset based on the argument N
        if self.N > 0:
            self.dataset = self.dataset.select(range(self.N))

        print(f"Loaded {len(self.dataset):,} examples from {self.__class__.__name__} dataset")

        self.timer = Timer()

    def validate_args(self):
        """Validate the arguments passed to the dataset"""
        assert self.return_sequence in [
            "full",
            "partial-random",
            "partial-response",
        ], f"Invalid return_sequence. Got {self.return_sequence}. Must be one of ['full', 'partial-random', 'partial-response']"
        assert self.split in ["train", "val", "test"], f"Invalid split. Got {self.split}. Must be one of ['train', 'val', 'test']"
        assert not self.use_chat_template or self.tokenizer.chat_template is not None, "Tokenizer does not have chat_template"
        assert self.categories in [
            "all",
            "medical_qa",
            "not_medical_qa",
        ], f"Invalid categories. Got {self.categories}. Must be one of ['all', 'medical_qa', 'not_medical_qa']"
        if self.debug:
            rich.print(f"[bold red]WARNING: DEBUG Mode on for dataset {self.__class__.__name__}[/bold red]")

    def load_categories(self):
        """Load the categories from the categories.txt file that is located in the root directory of the dataset"""
        # path = os.path.join(self.root, "categories.txt")
        # with open(path, "r") as f:
        #     categories = f.read().splitlines()
        # return [c.strip() for c in categories]
        # Search paths
        search_paths = [
            os.path.join(self.root, "categories.txt"),
            os.path.join(os.path.dirname(self.root), "categories.txt"),
        ]

        # Try to find the file in the search paths
        for path in search_paths:
            if os.path.exists(path):
                with open(path, "r") as f:
                    categories = f.read().splitlines()
                return [c.strip() for c in categories]

        # Raise an error if the file is not found
        raise FileNotFoundError("categories.txt not found in self.root or its parent directory.")

    def set_system_prompt(self) -> str:
        return "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."

    def load_dataset(self, dataset_name_or_path):
        # decide on dataset type
        builder = "plain_text"
        split = {"train": "train", "val": "train", "test": "validation"}[self.split]
        # load the raw dataset
        dataset = load_dataset(dataset_name_or_path, builder, split=split, trust_remote_code=False, cache_dir=self.root)

        # splitting
        self.random_seed = 23633
        dataset = dataset.shuffle(seed=self.random_seed)
        if self.split == "train":
            dataset = dataset.select(range(85000))
        elif self.split == "val":
            # take all the ones after
            dataset = dataset.select(range(85000, len(dataset)))

        if self.debug:
            dataset = dataset.select(range(min(self.N_DEBUG, len(dataset))))
        return dataset

    def get_cache_file_name(self):
        if self.sequence_prefix is None:
            prefix_str = "_NoPrefix"
        else:
            # the string might be too long for linux file name. We hash it to be sure.
            prefix_str = "_" + hashlib.sha256(self.sequence_prefix.encode()).hexdigest()[:16]
        return f"{self.root}/caches/squad_{self.split}_{f'debug{self.N_DEBUG}_' if self.debug else ''}{'_'.join(self.tokenizer.name_or_path.split('/')).replace('@', '_')}_{'chatTemplate' if self.use_chat_template else 'noChatTemplate'}_{self.training_task}_{f'splitSeed{self.random_seed}'}_maxLength{self.truncate_sequence_to}{prefix_str}.arrow"

    def process_data(self, dataset):
        # Process and tokenize the dataset
        os.makedirs(self.root + "/caches", exist_ok=True)
        cache_file_name = self.get_cache_file_name()

        return dataset.map(
            self.process_and_tokenize_function,
            remove_columns=dataset.column_names,
            cache_file_name=cache_file_name,
            num_proc=1,
        )

    @staticmethod
    def _truncate_sequence(tokenizer, input_text, answer, truncate_sequence_to):
        # Tokenize input_text and answer separately
        input_tokens = tokenizer.encode(input_text, add_special_tokens=False)
        transition_and_answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        max_input_length = truncate_sequence_to - len(transition_and_answer_tokens) - 2  # 2 for BOS and EOS

        # Truncate input_text tokens if necessary
        if len(input_tokens) > max_input_length:
            input_tokens = input_tokens[:max_input_length]

        # Reconstruct the truncated input_text (clumsy but safe)
        truncated_input_text = tokenizer.decode(input_tokens)

        return truncated_input_text

    def process_and_tokenize_function(self, example):
        contexts = example["context"]
        question = example["question"]
        answer_lengths = [len(a) for a in example["answers"]["text"]]
        answer = example["answers"]["text"][answer_lengths.index(max(answer_lengths))]

        if self.use_chat_template:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Question: {question}\nContext: {contexts}"},
                {"role": "assistant", "content": answer},
            ]
            tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", truncation=True)
            prompt_tokenized = self.tokenizer.apply_chat_template(messages[:-1], return_tensors="pt", add_generation_prompt=True)
            raise NotImplementedError("Chat template not implemented/validated yet")
        else:
            if self.sequence_prefix is not None:
                input_text = f"{self.sequence_prefix}\n\nQuestion: {question}\n\nContext: {contexts}"
            else:
                input_text = f"Question: {question}\n\nContext: {contexts}"
            transition_text = "\n\nAnswer: "

            # Truncate input_text if necessary
            truncated_input_text = self._truncate_sequence(self.tokenizer, input_text, answer, self.truncate_sequence_to)

            ###### Get the length of the prompt
            # Combine truncated input_text, transition and answer
            input_and_transition_tokens = self.tokenizer(truncated_input_text + transition_text)
            input_and_transition_length = len(input_and_transition_tokens["input_ids"])

            ###### Tokenize the combined text
            combined_text = truncated_input_text + transition_text + answer
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
            "id": example["id"],
            # "id_dec": torch.tensor([self.id_hex_2_decimal(example["id"])]),
            "title": example["title"],
            "category": torch.tensor([self.category2idx[example["title"]]]),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "length": len(input_ids),
            "n_token_prompt": torch.tensor([input_and_transition_length]),
            "len_prompt_text": torch.tensor([len(truncated_input_text + transition_text)]),
            "n_token_response": len(input_ids) - input_and_transition_length,
        }

    @staticmethod
    def id_hex_2_decimal(x: str) -> int:
        return int(x, 16)

    @staticmethod
    def id_decimal_2_hex(x: int) -> str:
        return hex(x)[2:]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        start_time = time.time()
        example = self.dataset[idx]

        return_dict = {
            "input_ids": example["input_ids"],
            # "attention_mask": example["attention_mask"],
            "n_token_prompt": example["n_token_prompt"],
            "len_prompt_text": example["len_prompt_text"],
            "idx": torch.tensor([idx]),
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

        self.timer.append(time.time() - start_time)

        for k, v in return_dict.items():
            if isinstance(v, list):
                return_dict[k] = torch.tensor(v)

        return return_dict

    @staticmethod
    def get_dataset_config_name(cfg) -> str:
        return f"{cfg.data.name}_{cfg.data.categories}"


class SQuADWithGeneratedResponsesDataset(SQuADDataset):
    N_DEBUG: int = 256

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        root: str,
        generated_output_path: str,
        keep_judge_decision: str,
        N: int,
        split: str,
        debug: bool,
        categories: str = "all",
        return_labels: bool = True,
        return_sequence: str = "full",
        truncate_sequence_to: int = None,
        min_response_length: int = 0,
        sequence_prefix: str = None,  # only used for loading the original dataset
        use_chat_template: bool = False,
        training_task: str = "seq2seq",
    ):
        """SQuAD dataset with y generated by model. This class loads the dataset from huggingface as commonly done.
        It then uses the output of the `certified_inference.py` script to overwrite the answers in the dataset with the
        generated answers. Further, it may utalize the judge decision to filter out samples that do not match the intended domain.

        Args:
            tokenizer (AutoTokenizer): A Huggingface tokenizer from transformers library. Used to tokenize the data.
            root (str): The root directory where the data is stored. e.g. "/data/pubmedqa".
            generated_output_path (str): The path to the generated output json file. This should be generated by `certified_inference.py`.
            keep_judge_decision (str): If set to None, all loaded samples are kept. If set to <decision> (see utils.judge.JudgeDecision),
                only those samples are kept. This can ensure cleaner in- and out-of-domain datasets.
            N (int): Number of datapoints to subset the dataset to. If N<=0, then the entire dataset is used.
            split (str): The split of the dataset. Must be one of ["train", "val", "test"].
            debug (bool): If True, only a subset of the dataset is used for debugging purposes.
            return_sequence (str, optional): The type of sequence to return. Must be one of ["full", "partial-random", "partial-response"]. Defaults to "full".
            truncate_sequence_to (int, optional): The maximum length of the sequence. Defaults to None.
            use_chat_template (bool, optional): If True, the tokenizer's chat_template is used to tokenize the data. Defaults to False.
            training_task (str, optional): The training task. Must be one of ["seq2seq", "lm"]. Defaults to "seq2seq".
        """
        self.tokenizer = tokenizer
        self.N = N
        self.split = split
        self.debug = debug
        self.dataset_num_token = None
        self.return_labels = return_labels
        self.return_sequence = return_sequence
        self.truncate_sequence_to = truncate_sequence_to
        self.min_response_length = min_response_length
        self.sequence_prefix = sequence_prefix
        self.use_chat_template = use_chat_template
        self.system_prompt = self.set_system_prompt()
        self.training_task = training_task
        self.categories = categories
        self.generated_output_path = generated_output_path
        self.base_model = generated_output_path.split("/")[-2]
        self.root = os.path.join(root, self.base_model)
        self.keep_judge_decision = keep_judge_decision

        if self.keep_judge_decision in ["none", "None", "all", "All"]:
            self.keep_judge_decision = None

        self.validate_args()

        # Load from text file a list of all possibel SQuAD categories
        self.idx2category = self.load_categories()
        self.category2idx = {c: i for i, c in enumerate(self.idx2category)}

        # Loading the official hf dataset
        dataset = self.load_dataset("rajpurkar/squad")

        # subset based on categories
        if categories == "medical_qa":
            dataset = dataset.filter(lambda x: x["title"] in MEDICAL_QA_CATEGORIES)
        elif categories == "not_medical_qa":
            dataset = dataset.filter(lambda x: x["title"] not in MEDICAL_QA_CATEGORIES)

        # Get number of response tokens in the dataset
        dataset = self.add_response_length_to_dataset(dataset)

        # capture the reponse lengthts and filter out samples with response length < min_response_length
        response_lengths = SequenceLengths()
        response_lengths.extend([x.as_py() for x in list(dataset.data["n_token_response"])])
        if min_response_length > 0:
            original_length = len(dataset)
            dataset = dataset.filter(lambda x: x["n_token_response"] > min_response_length)
            ddp.print(
                f"Filtered out {original_length - len(dataset):,} samples with response length < {min_response_length}. Remaining: {len(dataset):,}"
            )
            assert len(dataset) > 0, "No samples left after filtering for min_response_length"

        # subset based on the argument N
        if self.N > 0:
            dataset = dataset.select(range(self.N))

        # Overwrite the dataset with the generated samples
        dataset = self.overwrite_with_generated_samples(dataset, generated_output_path)
        original_length = len(dataset)
        dataset = dataset.filter(lambda x: x["keep_sample"])
        ddp.print(f"Filtered out {original_length - len(dataset)} samples that did not match the generated samples")

        # tokenize the dataset
        self.dataset = self.process_data(dataset)

        # capture the sequence lengths
        self.seq_lengths = SequenceLengths()
        self.seq_lengths.extend([x.as_py() for x in list(self.dataset.data["length"])])
        ddp.print(
            f"Min sequence length: {self.seq_lengths.min()} | Mean sequence length: {self.seq_lengths.mean():.2f} | Max sequence length: {self.seq_lengths.max()}"
        )

        ddp.print(f"Loaded {len(self.dataset):,} examples from {self.__class__.__name__} dataset")

        self.timer = Timer()

    def add_response_length_to_dataset(self, dataset: Dataset) -> Dataset:
        # Get number of response tokens in the dataset by loading the original dataset. Not clean but works.
        dataset_loaded = SQuADDataset(
            tokenizer=self.tokenizer,
            root=self.root,
            N=-1,
            split=self.split,
            debug=self.debug,
            categories=self.categories,
            return_labels=self.return_labels,
            return_sequence=self.return_sequence,
            min_response_length=0,
            sequence_prefix=self.sequence_prefix,
            truncate_sequence_to=self.truncate_sequence_to,
            use_chat_template=self.use_chat_template,
            training_task=self.training_task,
        )
        dataset = dataset.add_column("n_token_response", dataset_loaded.dataset["n_token_response"])
        dataset = dataset.add_column("length", dataset_loaded.dataset["length"])
        return dataset

    def get_cache_file_name(self) -> str:
        """Constructs cache file name based on the dataset configuration that is applied before the dataset is tokenized"""
        return f"{self.root}/caches/squad_{self.split}_{f'debug{self.N_DEBUG}_' if self.debug else ''}{self.base_model}_{self.N}_{'_'.join(self.tokenizer.name_or_path.split('/')).replace('@', '_')}_{'chatTemplate' if self.use_chat_template else 'noChatTemplate'}_{self.training_task}_{f'splitSeed{self.random_seed}'}_maxLength{self.truncate_sequence_to}.arrow"

    def verify_loaded_generation(self, loaded_results: List[Dict[str, Any]], loaded_config: List[Dict[str, Any]], dataset: Dataset) -> None:
        assert len(loaded_results) == len(dataset), (
            f"Length of generated samples ({len(loaded_results)}) does not match length of dataset ({len(dataset)})"
        )
        assert loaded_config["data"]["use_chat_template"] == self.use_chat_template, (
            f"Use chat template does not match. Got {loaded_config['data']['use_chat_template']} but expected {self.use_chat_template}"
        )
        assert loaded_config["data"]["categories"] == self.categories, (
            f"Categories do not match. Got {loaded_config['data']['categories']} but expected {self.categories}"
        )
        # assert (
        #     loaded_config["data"]["split"] == self.split
        # ), f"Split does not match. Got {loaded_config['data']['split']} but expected {self.split}"
        assert loaded_config["data"]["debug"] == self.debug, (
            f"Debug does not match. Got {loaded_config['data']['debug']} but expected {self.debug}"
        )
        # TODO assuming test set for now
        assert loaded_config["data"]["test_size"] == self.N, f"N does not match. Got {loaded_config['data']['N']} but expected {self.N}"
        assert self.split == "test", f"Expected split to be 'test' but got {self.split}"
        assert loaded_config["data"]["return_labels"] == self.return_labels, (
            f"Return labels does not match. Got {loaded_config['data']['return_labels']} but expected {self.return_labels}"
        )
        assert loaded_config["data"]["name"] == "squad", f"Expected dataset name to be 'squad' but got {loaded_config['data']['name']}"

    def overwrite_with_generated_samples(self, dataset: Dataset, model_generation_path: str):
        # Load the generated samples from json
        with open(model_generation_path, "r") as f:
            loaded_inference_output = json.load(f)
        loaded_config = loaded_inference_output["config"]
        loaded_results = loaded_inference_output["results"]

        # Check if original config matches the current config in relevant fields
        self.verify_loaded_generation(loaded_results, loaded_config, dataset)

        # add columns to the dataset
        dataset = dataset.add_column("prompt_text", [x["prompt_text"] for x in loaded_results])
        dataset = dataset.add_column("generated_text", [x["generated_text"] for x in loaded_results])
        dataset = dataset.add_column("decision", [x["decision"] for x in loaded_results])

        # Overwrite the dataset with the generated samples
        return dataset.map(
            self.overwrite_single_sample,
            keep_in_memory=True,
            num_proc=4,
            desc="Merging generated samples with dataset",
        )

    def overwrite_single_sample(self, example):
        keep_sample = True

        # check whether the samples match. use beginning of prompt (not exact but ok)
        if self.sequence_prefix is not None:
            ground_truth_question = f"{self.sequence_prefix}\n\nQuestion: {example['question']}"
        else:
            ground_truth_question = f"Question: {example['question']}\n"

        loaded_prompt = example["prompt_text"]
        min_length = min(len(ground_truth_question), len(loaded_prompt))
        if not ground_truth_question[:min_length] == loaded_prompt[:min_length]:
            keep_sample = False

        # check if the judge decision indicates to keep the sample
        if self.keep_judge_decision is not None:
            keep_sample = keep_sample and (self.keep_judge_decision == example["decision"])

        # overwrite the answers
        example["answers"]["text"] = [example["generated_text"]]
        example["keep_sample"] = keep_sample

        return example
