import random
import time
from typing import Dict, List, Union

import rich
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    TrainerCallback,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import evaluate
import wandb

from utils.general import Timer, verify_prompt_length_argument


DataCollator = Union[DataCollatorForLanguageModeling, DataCollatorForSeq2Seq]
AutoModel = Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]


class PrintExamplesCallback(TrainerCallback):
    """Prints decoded examples from the training and evaluation set during evaluation."""

    def __init__(
        self,
        run,
        tokenizer,
        text_generator,
        num_samples: int = 3,
        to_console: bool = True,
        to_wandb: bool = True,
    ):
        self.run = run
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.to_console = to_console
        self.to_wandb = to_wandb
        self.text_generator = text_generator

        self.wandb_cols = ["Dataset", "Index", "Epoch", "Step", "EvalLoss", "In", "In+Out", "Target"]

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return

        samples = self.text_generator(args, state, control, **kwargs)

        train_samples = self.decode_samples(samples["train_samples"], True, state.epoch, state.global_step, state.log_history[-1]["eval_loss"])
        eval_samples = self.decode_samples(samples["eval_samples"], False, state.epoch, state.global_step, state.log_history[-1]["eval_loss"])

        train_samples = train_samples[: self.num_samples]
        eval_samples = eval_samples[: self.num_samples]

        if self.to_console:
            self.print_to_console(train_samples, eval_samples)
        if self.to_wandb:
            self.log_to_wandb(train_samples, eval_samples)

    def decode_samples(self, samples, is_train: bool, epoch: int, step: int, loss: float):
        rows = []

        for idx, batch in enumerate(samples):
            input_decoded = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            response_decoded = self.tokenizer.batch_decode(batch["predicted"], skip_special_tokens=True)
            target_output_decoded = self.tokenizer.batch_decode(batch["target"], skip_special_tokens=True)

            for i in range(len(input_decoded)):
                rows.append(
                    [
                        "train" if is_train else "val",
                        batch["idx"][i].item(),
                        epoch,
                        step,
                        loss,
                        input_decoded[i],
                        response_decoded[i],
                        target_output_decoded[i],
                    ]
                )
        return rows

    def print_to_console(self, train_samples, eval_samples):
        for dataset, index, epoch, step, loss, prompt_decoded, response_decoded, target_output_decoded in train_samples:
            print(f"train:{index+1:06d} in='{prompt_decoded}' in+out='{response_decoded}' target='{target_output_decoded}'")
        for dataset, index, epoch, step, loss, prompt_decoded, response_decoded, target_output_decoded in eval_samples:
            print(f"eval:{index+1:06d} in='{prompt_decoded}' in+out='{response_decoded}' target='{target_output_decoded}'")

    def log_to_wandb(self, train_samples, eval_samples):
        print("\nLogging to wandb...", end=" ")
        train_table = wandb.Table(columns=self.wandb_cols, data=train_samples)
        eval_table = wandb.Table(columns=self.wandb_cols, data=eval_samples)
        wandb.log({"train/examples": train_table, "eval/examples": eval_table})
        print("Done.\n")


class TextGenerationMetrics(TrainerCallback):
    """Prints decoded examples from the training and evaluation set during evaluation."""

    def __init__(
        self,
        run,
        tokenizer,
        text_generator,
        compute_rouge: bool = True,
        compute_bleu: bool = True,
        compute_bertscore: bool = True,
    ):
        self.run = run
        self.tokenizer = tokenizer
        self.text_generator = text_generator

        self.rouge_metric = evaluate.load("rouge") if compute_rouge else None
        self.bleu_metric = evaluate.load("bleu") if compute_bleu else None
        self.bertscore_metric = evaluate.load("bertscore") if compute_bertscore else None

    def decode_samples(self, samples):
        responses = []
        labels = []

        for idx, batch in enumerate(samples):
            input_decoded = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            response_decoded = self.tokenizer.batch_decode(batch["predicted"], skip_special_tokens=True)
            target_output_decoded = self.tokenizer.batch_decode(batch["target"], skip_special_tokens=True)

            for i in range(len(input_decoded)):
                responses.append(response_decoded[i][len(input_decoded[i]) :])
                labels.append(target_output_decoded[i])

        return responses, labels

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return

        samples = self.text_generator(args, state, control, **kwargs)

        responses, labels = self.decode_samples(samples["eval_samples"])

        # Calculate metrics
        metrics = {}
        if self.rouge_metric is not None:
            rouge = self.rouge_metric.compute(predictions=responses, references=labels)
            metrics |= rouge
        if self.bleu_metric is not None:
            try:
                bleu = self.bleu_metric.compute(predictions=responses, references=labels)
                metrics["bleu"] = bleu["bleu"]
            except:
                metrics["bleu"] = float("nan")
        if self.bertscore_metric is not None:
            bertscore = self.bertscore_metric.compute(predictions=responses, references=labels)
            metrics |= bertscore

        # log to wandb
        self.run.log({f"eval/{k}": v for k, v in metrics.items()})


class CachedTextGenerator:
    # This will be referenced in all other callbacks. It will generate the outputs for the model once per eval step.
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForSeq2SeqLM,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: DataCollator,
        num_samples: int,
        max_length: int,
        batch_size: int,
        prompt_length: str,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.num_samples = num_samples
        self.max_length = max_length
        self.batch_size = batch_size
        self.prompt_length = prompt_length

        assert self.num_samples <= len(
            self.train_dataset
        ), f"The number of samples must be less than the training dataset size. Got {self.num_samples} > {len(self.train_dataset)}"
        assert self.num_samples <= len(
            self.eval_dataset
        ), f"The number of samples must be less than the evaluation dataset size. Got {self.num_samples} > {len(self.eval_dataset)}"

        self.train_indices = random.sample(range(len(self.train_dataset)), self.num_samples)
        self.eval_indices = random.sample(range(len(self.eval_dataset)), self.num_samples)
        self.samples_step = -1

        self.verify_prompt_length_argument()

        self.reset()

    def verify_prompt_length_argument(self):
        assert verify_prompt_length_argument(self.prompt_length)

    def reset(self):
        self.train_dataloader = DataLoader(Subset(self.train_dataset, self.train_indices), collate_fn=lambda x: x, batch_size=self.batch_size)
        self.eval_dataloader = DataLoader(Subset(self.eval_dataset, self.eval_indices), collate_fn=lambda x: x, batch_size=self.batch_size)

        # intialise storage of samples.
        self.train_samples = []
        self.eval_samples = []

    def __call__(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """Returns a list of samples for the training and evaluation set.

        Args:
            args (TrainingArguments): HF Callback args object.
            state (TrainerState):
            control (TrainerControl): _description_

        Returns:
            Dict[str, List[Dict[str, torch.Tensor]]]: Returns a dict with training, evaluation samples and the current step.
                Each set of samples is a list of batches. Each batch is a hugginface-style dictionary with extra keys.
        """
        # if the global step has changed, generate new samples.
        if not state.global_step == self.samples_step:
            self.samples_step = state.global_step
            self.reset()
            self.generate()

        # return samples
        return {
            "train_samples": self.train_samples,
            "eval_samples": self.eval_samples,
            "step": self.samples_step,
        }

    def generate(self):
        for idx, batch in enumerate(self.train_dataloader):
            # print(f"Generating Text: Train {idx+1}/{len(self.train_dataloader)}")
            self.generate_text(batch, is_train=True)
        for idx, batch in enumerate(self.eval_dataloader):
            # print(f"Generating Text:  Eval {idx+1}/{len(self.eval_dataloader)}")
            self.generate_text(batch, is_train=False)

    def generate_text(self, batch, is_train):
        # split sequence into prompt and target
        for element in batch:
            input_ids = element["input_ids"]
            if self.prompt_length == "dataset":
                n_token_prompt = element["n_token_prompt"]
            elif self.prompt_length == "random":
                n_token_prompt = random.randint(1, len(input_ids))
            else:
                n_token_prompt = int(self.prompt_length)
            prompt = input_ids[:n_token_prompt]
            target_output = input_ids[n_token_prompt:]
            element["input_ids"] = prompt
            element["target"] = target_output  # if you call it labels, it gets truncated by hf.
            element["prompt_length"] = torch.tensor([n_token_prompt])  # the length used to cut the prompt

        # pad
        max_prompt_length = max([len(element["input_ids"]) for element in batch])
        max_target_length = max([len(element["target"]) for element in batch])
        for element in batch:
            # left pad the prompt
            element["input_ids"] = torch.cat(
                [torch.full((max_prompt_length - len(element["input_ids"]),), self.tokenizer.pad_token_id), element["input_ids"]]
            )
            # right pad the target
            element["target"] = torch.cat(
                [element["target"], torch.full((max_target_length - len(element["target"]),), self.tokenizer.pad_token_id)]
            )

        collated_batch = self.data_collator(batch).to(self.model.device)

        # Generate a response
        outputs = self.model.generate(
            collated_batch["input_ids"],
            attention_mask=torch.ones_like(collated_batch["input_ids"]),
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.max_length,
        )

        collated_batch["predicted"] = outputs

        collated_batch = {k: v.cpu() for k, v in collated_batch.items()}

        if is_train:
            self.train_samples.append(collated_batch)
        else:
            self.eval_samples.append(collated_batch)


class Seq2SeqMetricsCallback(TrainerCallback):
    """Prints decoded examples from the training and evaluation set during evaluation."""

    def __init__(
        self,
        run,
        tokenizer,
        name,
        train_dataset,
        eval_dataset,
        model,
        max_length: int = 100,
        num_examples: int = 50,
    ):
        self.run = run
        self.tokenizer = tokenizer
        self.name = name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.num_examples = num_examples
        self.model = model
        self.max_length = max_length

        self.train_indices = random.sample(range(len(self.train_dataset)), self.num_examples)
        self.eval_indices = random.sample(range(len(self.eval_dataset)), self.num_examples)

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return

        rich.print(
            f"\n[bold green]Run: {self.run.name} Epoch: {state.epoch:.3f} Step: {state.global_step:,} / {state.max_steps:,} (BS = {state.train_batch_size:,}) [/bold green]"
            + ("(SAVING STEP)" if state.global_step % state.save_steps == 0 else "")
        )
        self.train_corr, self.train_total = 0, 0
        self.eval_corr, self.eval_total = 0, 0

        for idx in self.train_indices:
            example = self.train_dataset[idx]
            self.check_decoded_example(idx, example, is_train=True)
        rich.print(f"\n[bold green]{self.name} Train Accuracy: {self.train_corr/self.train_total:.2f}[/bold green]\n")

        # Print examples from the evaluation dataset
        for idx in self.eval_indices:
            example = self.eval_dataset[idx]
            self.check_decoded_example(idx, example, is_train=False)
        rich.print(f"\n[bold green]{self.name} Eval Accuracy: {self.eval_corr/self.eval_total:.2f}[/bold green]\n")

        # log to wandb
        self.run.log(
            {
                f"train/Accuracy - {self.name}": self.train_corr / self.train_total,
                f"eval/Accuracy - {self.name}": self.eval_corr / self.eval_total,
            }
        )

    def check_decoded_example(self, index, example, is_train):
        # Assuming `input_ids` is a key in your dataset. Adjust as necessary.
        input_ids = example["input_ids"].to(self.model.device)

        # get prompt
        length_input = example["n_token_prompt"][0]  # int(torch.where(example["labels"] >= 0)[0][0])
        prompt = input_ids[:length_input]
        target_output = input_ids[length_input:]

        # Generate a response
        outputs = self.model.generate(
            prompt.unsqueeze(0),
            attention_mask=torch.ones_like(prompt.unsqueeze(0)),
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.max_length,
        )  # Adjust `max_length` as needed

        # Decode inputs and outputs
        target_output_decoded = self.tokenizer.decode(target_output)
        response_decoded = self.tokenizer.decode(outputs.squeeze().tolist())

        # check if the response is correct
        response_decoded = response_decoded[-len(target_output_decoded) :]
        is_correct = int(response_decoded == target_output_decoded)

        if is_train:
            self.train_total += 1
            self.train_corr += is_correct
        else:
            self.eval_total += 1
            self.eval_corr += is_correct


class TimerCallback(TrainerCallback):
    def __init__(self):
        self.timer = Timer()

        self.current_time = None

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # the data is loaded by this point. this is pure training time
        self.current_time = time.time()
        return super().on_step_begin(args, state, control, **kwargs)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.timer.append(time.time() - self.current_time)
        return super().on_step_end(args, state, control, **kwargs)

    # def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    # doesnt work. no access to trainer class.
    #     dataset = kwargs["train_dataloader"].dataset
    #     print(f"Data loading time per sample: {dataset.timer.mean():.6f}s")
    #     return super().on_epoch_end(args, state, control, **kwargs)


class StepDebugger(TrainerCallback):
    def __init__(self, step: int, data_collator: DataCollator):
        self.step = step
        self.data_collator = data_collator

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == self.step:
            self.data_collator.should_break = True

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        pass
