import json
import os
import time
from typing import Dict
import rich
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.general import Timer
from utils import ddp

from .utilities import SequenceLengths, ScoringDataExample

PUBMEDQA_SCORING_MARKER = [" yes", " no", " maybe"]
CORRECT_ANSWER_MAP = {"yes": 0, "no": 1, "maybe": 2}


class PubMedQADataset(Dataset):
    N_DEBUG: int = 256

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        root: str,
        N: int,
        split: str,
        debug: bool,
        return_labels: bool = True,
        return_sequence: str = "full",
        truncate_sequence_to: int = None,
        use_chat_template: bool = False,
        training_task: str = "seq2seq",
    ):
        """PubmedQA dataset

        Args:
            tokenizer (AutoTokenizer): A Huggingface tokenizer from transformers library. Used to tokenize the data.
            root (str): The root directory where the data is stored. e.g. "/data/pubmedqa".
            N (int): Number of datapoints to subset the dataset to. If N<=0, then the entire dataset is used.
            split (str): The split of the dataset. Must be one of ["train", "val", "test"].
            debug (bool): If True, only a subset of the dataset is used for debugging purposes.
            return_sequence (str, optional): The type of sequence to return. Must be one of ["full", "partial-random", "partial-response"]. Defaults to "full".
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
        self.truncate_sequence_to = truncate_sequence_to
        self.use_chat_template = use_chat_template
        self.system_prompt = self.set_system_prompt()
        self.training_task = training_task
        # self.system_prompt_tokens = self.tokenizer(self.system_prompt, truncation=True)
        # self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        self.validate_args()

        self.timer = Timer()

        # self.track_indices = torch.full((len(self.tokenizer),), 0)

        # Loading the official hf dataset
        dataset = self.load_dataset("qiaojin/PubMedQA")
        self.dataset = self.process_data(dataset)

        # capture the sequence lengths
        self.seq_lengths = SequenceLengths()
        self.seq_lengths.extend([x.as_py() for x in list(self.dataset.data["length"])])
        ddp.print(
            f"Min sequence length: {self.seq_lengths.min()} | Mean sequence length: {self.seq_lengths.mean():.2f} | Max sequence length: {self.seq_lengths.max()}"
        )

        # subset based on the argument N
        if self.N > 0:
            self.dataset = self.dataset.select(range(self.N))

        print(f"Loaded {len(self.dataset):,} examples from {self.__class__.__name__} dataset")

        self.__post_init__()

    def __post_init__(self):
        """Post initialization function. Can be used to set additional attributes or perform additional operations for each subclass."""
        pass

    def validate_args(self):
        # Check for invalid arguments
        assert self.return_sequence in [
            "full",
            "partial-random",
            "partial-response",
        ], f"Invalid return_sequence. Got {self.return_sequence}. Must be one of ['full', 'partial-random', 'partial-response']"
        assert self.split in ["train", "val", "test"], f"Invalid split. Got {self.split}. Must be one of ['train', 'val', 'test']"
        assert not self.use_chat_template or self.tokenizer.chat_template is not None, "Tokenizer does not have chat_template"
        if self.debug:
            rich.print(f"[bold red]WARNING: DEBUG Mode on for dataset {self.__class__.__name__}[/bold red]")

    def set_system_prompt(self) -> str:
        return "You are an expert and experienced from the healthcare and biomedical domain with extensive medical knowledge and practical experience. Your name is OpenBioLLM, and you were developed by Saama AI Labs. who's willing to help answer the user's query with explanation. In your explanation, leverage your deep medical expertise such as relevant anatomical structures, physiological processes, diagnostic criteria, treatment guidelines, or other pertinent medical concepts. Use precise medical terminology while still aiming to make the explanation clear and accessible to a general audience."

    def load_dataset(self, dataset_name_or_path):
        # decide on dataset type
        if self.split == "test":
            builder = "pqa_labeled"
        else:
            builder = "pqa_artificial"

        # load the raw dataset
        dataset = load_dataset(dataset_name_or_path, builder, split="train", trust_remote_code=True, cache_dir=self.root)

        # splitting
        self.random_seed = 23633
        dataset = dataset.shuffle(seed=self.random_seed)
        if self.split == "train":
            dataset = dataset.select(range(200000))
        elif self.split == "val":
            # take all the ones after
            dataset = dataset.select(range(200000, len(dataset)))

        if self.debug:
            dataset = dataset.select(range(min(self.N_DEBUG, len(dataset))))
        return dataset

    def get_cache_file_name(self):
        return f"{self.root}/caches/pubmedqa_{self.split}_{f'debug{self.N_DEBUG}_' if self.debug else ''}{'_'.join(self.tokenizer.name_or_path.split('/')).replace('@', '_')}_{'chatTemplate' if self.use_chat_template else 'noChatTemplate'}_{self.training_task}_{f'splitSeed{self.random_seed}'}_maxLength{self.truncate_sequence_to}.arrow"

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

    def process_and_tokenize_function(self, example):
        contexts = "\n".join(example["context"]["contexts"]) + "\n"
        question = example["question"]
        answer = example["long_answer"]

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
            input_text = f"Question: {question}\n\nContext: {contexts}"
            transition_text = "\n\nAnswer: "

            #######  Truncate input_text if necessary
            # Tokenize input_text and answer separately
            input_tokens = self.tokenizer.encode(input_text, add_special_tokens=False)
            transition_and_answer_tokens = self.tokenizer.encode(answer, add_special_tokens=False)
            max_input_length = self.truncate_sequence_to - len(transition_and_answer_tokens) - 2  # 2 for BOS and EOS

            # Truncate input_text tokens if necessary
            if len(input_tokens) > max_input_length:
                input_tokens = input_tokens[:max_input_length]

            # Reconstruct the truncated input_text (clumsy but safe)
            truncated_input_text = self.tokenizer.decode(input_tokens)

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
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "n_token_prompt": torch.tensor([input_and_transition_length]),
            "len_prompt_text": torch.tensor([len(truncated_input_text + transition_text)]),
            "length": len(input_ids),
        }

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
            # split after BOS token and 5 before the end of the sequence. to leave at least a few tokens to learn on.
            split_point = torch.randint(1, len(example["input_ids"]) - 5, (1,)).item()
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
            if isinstance(v, torch.Tensor):
                assert v.shape[0] <= self.truncate_sequence_to, (
                    f"Sequence is too long. v.shape[0]={v.shape[0]} > self.truncate_sequence_to={self.truncate_sequence_to}"
                )

        return return_dict

    @staticmethod
    def get_dataset_config_name(cfg) -> str:
        return f"{cfg.data.name}"


class PubMedQAWithGeneratedResponsesDataset(PubMedQADataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        root: str,
        generated_output_path: str,
        keep_judge_decision: str,
        N: int,
        split: str,
        debug: bool,
        return_labels: bool = True,
        return_sequence: str = "full",
        truncate_sequence_to: int = None,
        use_chat_template: bool = False,
        training_task: str = "seq2seq",
    ):
        """PubmedQA dataset with y generated by model. This class loads the dataset from huggingface as commonly done.
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
            use_chat_template (bool, optional): If True, the tokenizer's chat_template is used to tokenize the data. Defaults to False.
            training_task (str, optional): The training task. Must be one of ["seq2seq", "lm"]. Defaults to "seq2seq".
            truncate_sequence_to (int, optional): The maximum length of the sequence. Defaults to None.
        """
        self.tokenizer = tokenizer
        self.N = N
        self.split = split
        self.debug = debug
        self.dataset_num_token = None
        self.return_labels = return_labels
        self.return_sequence = return_sequence
        self.truncate_sequence_to = truncate_sequence_to
        self.use_chat_template = use_chat_template
        self.system_prompt = self.set_system_prompt()
        self.training_task = training_task
        self.generated_output_path = generated_output_path
        self.base_model = generated_output_path.split("/")[-2]
        self.root = os.path.join(root, self.base_model)
        self.keep_judge_decision = keep_judge_decision

        if self.keep_judge_decision in ["none", "None", "all", "All", "ALL"]:
            self.keep_judge_decision = None

        self.validate_args()

        # Loading the official hf dataset
        dataset = self.load_dataset("qiaojin/PubMedQA")

        # overwrite answers with generated answers
        dataset = self.overwrite_with_generated_samples(dataset, generated_output_path)
        original_length = len(dataset)
        dataset = dataset.filter(lambda x: x["keep_sample"])
        print(f"Filtered out {original_length - len(dataset)} samples that did not match the generated samples")

        # tokenize the dataset
        self.dataset = self.process_data(dataset)

        # capture the sequence lengths
        self.seq_lengths = SequenceLengths()
        self.seq_lengths.extend([x.as_py() for x in list(self.dataset.data["length"])])
        ddp.print(
            f"Min sequence length: {self.seq_lengths.min()} | Mean sequence length: {self.seq_lengths.mean():.2f} | Max sequence length: {self.seq_lengths.max()}"
        )

        # subset based on the argument N
        if self.N > 0:
            self.dataset = self.dataset.select(range(self.N))

        print(f"Loaded {len(self.dataset):,} examples from {self.__class__.__name__} dataset")

        # initialize timer to time the data loading
        self.timer = Timer()

    def get_cache_file_name(self) -> str:
        return f"{self.root}/caches/pubmedqa_{self.split}_{f'debug{self.N_DEBUG}_' if self.debug else ''}{self.base_model}_{'_'.join(self.tokenizer.name_or_path.split('/')).replace('@', '_')}_{'chatTemplate' if self.use_chat_template else 'noChatTemplate'}_{self.training_task}_{f'splitSeed{self.random_seed}'}_maxLength{self.truncate_sequence_to}.arrow"

    def overwrite_with_generated_samples(self, dataset: Dataset, model_generation_path: str) -> Dataset:
        # Load the generated samples from json
        with open(model_generation_path, "r") as f:
            loaded_inference_output = json.load(f)
        loaded_config = loaded_inference_output["config"]
        loaded_results = loaded_inference_output["results"]

        if len(loaded_results) < len(dataset):
            ddp.pprint(
                f"[bold yellow]WARNING[/bold yellow]: Loaded results ({len(loaded_results)}) are less than the dataset ({len(dataset)}). Subsetting"
            )
            dataset = dataset.select(range(len(loaded_results)))

        # Check if original config matches the current config
        # TODO add more checks
        assert len(loaded_results) == len(dataset), (
            f"Length of generated samples ({len(loaded_results)}) does not match length of dataset ({len(dataset)})"
        )

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

    def overwrite_single_sample(self, example: Dict) -> Dict:
        keep_sample = True

        # check whether the samples match. use beginning of prompt (not exact but ok)
        ground_truth_question = f"Question: {example['question']}"
        loaded_prompt = example["prompt_text"]
        min_length = min(len(ground_truth_question), len(loaded_prompt))
        if not ground_truth_question[:min_length] == loaded_prompt[:min_length]:
            keep_sample = False

        # check if the judge decision indicates to keep the sample
        if self.keep_judge_decision is not None:
            keep_sample = keep_sample and (self.keep_judge_decision == example["decision"])

        example["long_answer"] = example["generated_text"]
        example["keep_sample"] = keep_sample

        return example


class PubMedQAScoringDataset(PubMedQADataset):
    # instruction = "Use the context provided below to answer the question with yes / no / maybe."  # the answer tokens here match the ones appended later for the choices
    instruction = None

    # def __post_init__(self):
    #     ret = super().__post_init__()
    #     # self.instruction = ""
    #     return ret

    @property
    def use_instruction(self) -> bool:
        return self.instruction is not None

    def get_cache_file_name(self) -> str:
        return f"{self.root}/caches/pubmedqa_scoring_{self.split}_{f'debug{self.N_DEBUG}_' if self.debug else ''}_{'_'.join(self.tokenizer.name_or_path.split('/')).replace('@', '_')}_{'chatTemplate' if self.use_chat_template else 'noChatTemplate'}_{'instruction' if self.use_instruction else 'no_instruction'}.arrow"

    def process_and_tokenize_function(self, example):
        contexts = "\n".join(example["context"]["contexts"]) + "\n"
        question = example["question"]
        # answer = example["long_answer"]
        correct_answer = example["final_decision"]

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
            instruction = f"Instruction: {self.instruction}" if self.instruction is not None else ""
            input_text = f"{instruction}\n\nQuestion: {question}\n\nContext: {contexts}"
            transition_text = "\n\nAnswer:"

            ###### Tokenize the combined text
            combined_text = input_text + transition_text
            tokenized = self.tokenizer(combined_text, return_tensors="pt")
            input_and_transition_length = len(tokenized["input_ids"][0])

        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        labels = input_ids.clone()
        if self.training_task == "seq2seq":
            labels[:input_and_transition_length] = -100

        # assert sequence charactereistics
        assert input_ids.shape == attention_mask.shape, f"input_ids.shape={input_ids.shape} != attention_mask.shape={attention_mask.shape}"
        assert input_ids[0] == self.tokenizer.bos_token_id, f"input_ids[0]={input_ids[0]} != bos_token_id={self.tokenizer.bos_token_id}"

        # # due to the possible imprecision of the tokenization, we truncate the sequence to truncate_sequence_to in the response:
        # if self.truncate_sequence_to is not None:
        #     input_ids = input_ids[: self.truncate_sequence_to]
        #     attention_mask = attention_mask[: self.truncate_sequence_to]
        #     labels = labels[: self.truncate_sequence_to]

        # assert input_ids.shape[0] <= self.truncate_sequence_to, (
        #     f"input_ids.shape[0]={input_ids.shape[0]} > truncate_sequence_to={self.truncate_sequence_to}"
        # )

        return {
            "question": question,
            "subject": "N/A",
            "choices": ["yes", "no", "maybe"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "n_token_prompt": torch.tensor([input_and_transition_length]),
            "len_prompt_text": torch.tensor([len(combined_text)]),
            "length": len(input_ids),
            "correct_answer_id": torch.tensor([CORRECT_ANSWER_MAP[correct_answer]]),
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        start_time = time.time()
        example = self.dataset[idx]
        return_dict = {"idx": torch.tensor([idx]), **example}

        return_dict["subject_id"] = torch.tensor([-1])
        return_dict["n_shot_examples"] = []  # PubmedQA is 0-shot

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
