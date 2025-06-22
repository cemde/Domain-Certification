# ruff: noqa: E402
import os
from typing import Any, Tuple

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)
import wandb

import utils
from utils.collators import DataCollatorForLanguageModelingTimed, DataCollatorForSeq2SeqTimed
from utils.data import (
    Dataset,
    TaskCollectionDataset,
    TaskDatasetTraining,
    TinyShakespeareDataset,
    PubMedQADataset,
    PubMedQAWithGeneratedResponsesDataset,
    MMLUQADataset,
    mmlu_categories,
)
from utils.models import GPT_SIZES

torch.set_printoptions(linewidth=220, edgeitems=10)


cluster = utils.cluster.ClusterManager()


def get_data(cfg: DictConfig, tokenizer: AutoTokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    """Get the training, validation and test datasets.

    Args:
        cfg (DictConfig): The configuration.
        tokenizer (AutoTokenizer): The tokenizer.

    Returns:
        Tuple[Dataset, Dataset, Dataset]: The training, validation and test datasets.
    """
    if cfg.data.name == "task_data":
        kwargs = {
            "modality": cfg.data.modality,
            "task": cfg.data.task,
            "max_seq_length": cfg.data.max_sequence_length,
            "num_int": cfg.data.num_int,
            "num_char": cfg.data.num_char,
            "constant_seq_length": cfg.data.constant_sequence_length,
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "TaskData"),
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
        }

        train_dataset = TaskDatasetTraining(N=cfg.data.train_size, split="train", **kwargs)
        val_dataset = TaskDatasetTraining(N=cfg.data.val_size, split="val", **kwargs)
        test_dataset = None

    elif cfg.data.name == "task_data_collection":
        kwargs = {
            "modality": cfg.data.modality,
            "max_seq_length": cfg.data.max_sequence_length,
            "num_int": cfg.data.num_int,
            "num_char": cfg.data.num_char,
            "constant_seq_length": cfg.data.constant_sequence_length,
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "TaskData"),
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
        }

        train_datasets = []
        val_datasets = []

        for task in cfg.data.task.split(","):
            kwargs["task"] = task
            train_datasets.append(TaskDatasetTraining(N=cfg.data.train_size, split="train", **kwargs))
            val_datasets.append(TaskDatasetTraining(N=cfg.data.val_size, split="val", **kwargs))

        train_dataset = TaskCollectionDataset(N=cfg.data.train_size, datasets=train_datasets)
        val_dataset = TaskCollectionDataset(N=cfg.data.val_size, datasets=val_datasets)
        test_dataset = None

    elif cfg.data.name == "shakespeare":
        kwargs = {
            "tokenizer": tokenizer,
            "context_length": cfg.model.context_length,
            "root": os.path.join(cluster.data_dir, "TinyShakespeare"),
            "return_sequence": cfg.data.return_sequence,
            "prepend_bos": cfg.data.prepend_bos,
            "append_eos": cfg.data.append_eos,
        }
        train_dataset = TinyShakespeareDataset(N=cfg.data.train_size, split="train", **kwargs)
        val_dataset = TinyShakespeareDataset(N=cfg.data.train_size, split="val", **kwargs)
        test_dataset = None

    elif cfg.data.name == "pubmedqa":
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "PubMedQA"),
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "debug": cfg.data.debug,
            "truncate_sequence_to": cfg.data.truncate_sequence_to,
            "training_task": cfg.training.task,
            "use_chat_template": cfg.data.use_chat_template,
        }
        train_dataset = PubMedQADataset(N=cfg.data.train_size, split="train", **kwargs)
        val_dataset = PubMedQADataset(N=cfg.data.val_size, split="val", **kwargs)
        test_dataset = None

        assert not cfg.training.group_by_length, "Strange bug with pubmedqa and group_by_length. Disabling for now."

    elif cfg.data.name == "pubmedqa_generated":
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "PubMedQAWithGeneratedResponses"),
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "debug": cfg.data.debug,
            "truncate_sequence_to": cfg.data.truncate_sequence_to,
            "training_task": cfg.training.task,
            "use_chat_template": cfg.data.use_chat_template,
            "keep_judge_decision": cfg.data.keep_judge_decision,
        }
        train_dataset = PubMedQAWithGeneratedResponsesDataset(
            N=cfg.data.train_size,
            split="train",
            generated_output_path=os.path.join(cluster.artifact_dir, cfg.data.generated_output_paths.train),
            **kwargs,
        )
        val_dataset = PubMedQAWithGeneratedResponsesDataset(
            N=cfg.data.val_size,
            split="val",
            generated_output_path=os.path.join(cluster.artifact_dir, cfg.data.generated_output_paths.val),
            **kwargs,
        )
        test_dataset = None

        assert not cfg.training.group_by_length, "Strange bug with pubmedqa and group_by_length. Disabling for now."

    elif cfg.data.name == "mmlu":
        kwargs = {
            "tokenizer": tokenizer,
            "root": os.path.join(cluster.data_dir, "MMLU"),
            "debug": cfg.data.debug,
            "categories": mmlu_categories(cfg.data.categories),
            "return_labels": cfg.data.return_labels,
            "return_sequence": cfg.data.return_sequence,
            "min_response_length": cfg.data.min_response_length,
            "return_answer": cfg.data.return_answer,
            "truncate_sequence_to": cfg.data.truncate_sequence_to,
            "training_task": cfg.training.task,
            "use_chat_template": cfg.data.use_chat_template,
            "n_shot": cfg.data.n_shot,
        }
        # for validation, we always use partial-response, as this is the sequence we care about
        val_kwargs = kwargs | {"return_sequence": "partial-response", "return_answer": "correct", "min_response_length": 1}

        train_dataset = MMLUQADataset(N=cfg.data.train_size, split="train", **kwargs)
        val_dataset = MMLUQADataset(N=cfg.data.val_size, split="val", **val_kwargs)
        test_dataset = None
    else:
        raise ValueError(f"Unknown dataset {cfg.data.name}.")

    # print some examples
    if cfg.log.print_data_examples_on_start_up:
        utils.data.print_examples(train_dataset, "train", tokenizer)
        utils.data.print_examples(val_dataset, "val", tokenizer)
        if test_dataset is not None:
            utils.data.print_examples(test_dataset, "test", tokenizer)
        else:
            utils.ddp.pprint("No test dataset to print examples from.")
    return train_dataset, val_dataset, test_dataset


def get_model(cfg: DictConfig, tokenizer: AutoTokenizer, model_dir: str, device_map: Any) -> nn.Module:
    if cfg.model.architecture in GPT_SIZES:
        # Small custom GPT-2 model
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_ctx=cfg.model.context_length,
            resid_pdrop=cfg.model.residual_dropout,
            embd_pdrop=cfg.model.embedding_dropout,
            attn_pdrop=cfg.model.attention_dropout,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            device_map=device_map,
            **GPT_SIZES[cfg.model.architecture],
        )
        model = GPT2LMHeadModel(config)

    elif cfg.model.architecture in ["openai-community/gpt2"]:
        config = AutoConfig.from_pretrained(cfg.model.architecture, device_map=device_map)
        model = AutoModelForCausalLM.from_pretrained(cfg.model.architecture, config=config, cache_dir=model_dir)
        # model = model.to(torch.bfloat16)  # doesn't it have to be lora then?

    elif cfg.model.architecture == "princeton-nlp/Sheared-LLaMA-1.3B":
        # LLaMA-2 pruned model: princeton-nlp/Sheared-LLaMA-1.3B
        bandb_config = BitsAndBytesConfig(load_in_4bit=True)
        config = AutoConfig.from_pretrained(cfg.model.architecture, quantization_config=bandb_config)
        model = AutoModelForCausalLM.from_pretrained(cfg.model.architecture, config=config, cache_dir=model_dir)
        model = model.to(torch.bfloat16)  # doesn't it have to be lora then?

    elif cfg.model.architecture == "llama-3-8b":
        raise NotImplementedError("LLaMA-3 8-bit model is not supported yet.")

    elif cfg.model.architecture == "google/gemma-2-2b":
        model = AutoModelForCausalLM.from_pretrained(cfg.model.architecture, cache_dir=model_dir)
        model = model.to(torch.bfloat16)  # doesn't it have to be lora then?

    elif cfg.model.source == "local":
        model = AutoModelForCausalLM.from_pretrained(os.path.join(model_dir, cfg.model.architecture), cache_dir=model_dir)

    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.architecture, cache_dir=model_dir)

    model_size = sum(t.numel() for t in model.parameters())
    utils.ddp.print(f"Model={cfg.model.architecture}. Size: {model_size:,} ({model_size / (1000**2):.2f}M) parameters.")

    # resize token embeddings if tokenizer was modified and a pad token was added
    if cfg.tokenizer.add_pad_token:
        model.resize_token_embeddings(len(tokenizer))

    return model


def get_tokenizer(cfg: DictConfig, tokenizer_dir: str) -> AutoTokenizer:
    if cfg.run.load_data_in_parallel:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # add path to tokenizer name if not from huggingface
    tokenizer_name = cfg.tokenizer.name if cfg.tokenizer.source == "hf" else os.path.join(tokenizer_dir, cfg.tokenizer.name)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=tokenizer_dir, padding_side=cfg.tokenizer.padding_side)
    if cfg.tokenizer.overwrite_pad_with_eos_token:
        assert cfg.tokenizer.padding_side == "right", "Padding side must be right if overwriting pad token with eos token."
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.tokenizer.add_pad_token:
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            utils.ddp.pprint("[bold yellow]WARNING: Added [PAD] token to tokenizer.")
        else:
            raise ValueError(f"Tokenizer already has a pad token: {tokenizer.pad_token}")
    return tokenizer


def validate_cfg(cfg: DictConfig, is_ddp: bool, world_size: int) -> None:
    """In an attempt to make the script more robust, this function validates the configuration.

    Args:
        cfg (DictConfig): The configuration.
        is_ddp (bool): Whether DDP is enabled.
        world_size (int): The number of GPUs.
    """
    if not cfg.data.return_labels:
        assert cfg.training.task == "causal", (
            f"If data not returning label, training must be 'causal'. Got task={cfg.training.task} and return_labels={cfg.data.return_labels}"
        )

    if cfg.training.task == "causal":
        assert not cfg.data.return_labels, (
            f"If training is 'causal', data must not return labels. This can result in a data collation error. Got task={cfg.training.task} and return_labels={cfg.data.return_labels}."
        )

    if cfg.model.load_weights_in_8_bit:
        raise NotImplementedError("Loading weights in 8-bit is not supported yet.")

    # overwriting pad token with eos token is only supported for seq2seq tasks and right padding
    if cfg.tokenizer.overwrite_pad_with_eos_token:
        # causal modelling while overwriting pad token, leads to never ending generation (never predicting EOS).
        assert cfg.training.task == "seq2seq", (
            f"Overwriting pad token with eos token is only supported for seq2seq tasks. Got task={cfg.training.task}"
        )
        # right padding is required for overwriting pad token with eos token. with left pedding the sentence ends before it starts.
        assert cfg.tokenizer.padding_side == "right", (
            f"Overwriting pad token with eos token is only supported for right padding. Got padding_side={cfg.tokenizer.padding_side}"
        )

    if cfg.data.name == "task_data":
        assert not (cfg.data.return_sequence == "partial-random" and cfg.training.task == "seq2seq"), (
            f"Partial-random sequence generation is only supported for causal training. Got {cfg.data.return_sequence=} and {cfg.training.task=}"
        )
        assert not (cfg.data.return_sequence == "partial-response" and cfg.training.task == "causal"), (
            f"Partial-response sequence generation is only supported for seq2seq training. Got {cfg.data.return_sequence=} and {cfg.training.task=}"
        )

    # handling batch size
    assert cfg.optim.overall_batch_size % world_size == 0, "Batch size must be divisible by number of GPUs."
    assert cfg.optim.overall_batch_size % cfg.optim.per_device_batch_size == 0, "Batch size must be divisible by per device batch size."
    assert cfg.optim.overall_batch_size / (cfg.optim.per_device_batch_size * world_size) >= 1, (
        "Overall batch size must be at least equal to number of devices times per device batch size."
    )

    if is_ddp:
        assert not cfg.run.compile, f"Compilation is not supported with DDP. Got {cfg.run.compile=} Torch Dynamo throws an exception."

    if cfg.training.mixed_precision and not utils.supports_bfloat16():
        utils.ddp.pprint("[bold yellow]Mixed Precision=True, but GPU does not support BF16. Can be instable.[/bold yellow]")

    # fix config types
    cfg.data.train_size = int(cfg.data.train_size)
    cfg.data.val_size = int(cfg.data.val_size)
    cfg.data.test_size = int(cfg.data.test_size)

    # freeze it so that it is not modified
    OmegaConf.set_readonly(cfg, True)


def setup_logging(cfg: DictConfig) -> Tuple[Any, str, str]:
    """Setup logging with Weights & Biases.

    Args:
        cfg (DictConfig): The configuration.

    Returns:
        Tuple[Any, str, str]: The wandb run, run name and experiment directory.
    """
    logging_mode = cfg.log.mode
    if cfg.run.debug:
        logging_mode = "disabled"
    if cluster.network == "offline" and logging_mode == "online":
        logging_mode = "offline"

    if utils.ddp.is_main_process():
        run = wandb.init(
            entity="<your_wandb_entity>",
            project="<your_wandb_project>",
            config={"hydra": OmegaConf.to_container(cfg)},
            job_type="train",
            tags=cfg.log.tags,
            dir=cluster.log_dir,
            mode=logging_mode,
        )
        experiment_dir = os.path.join(cluster.artifact_dir, "models", cfg.data.domain, run.name)
        os.makedirs(experiment_dir, exist_ok=True)
        run_name = run.name

        if cfg.log.print_config:
            print("=" * 80)
            print("CONFIGURATION")
            print("=" * 80)
            print(OmegaConf.to_yaml(cfg))
            print("=" * 80)

    else:
        run = None
        run_name = ""
        experiment_dir = ""

    return run, run_name, experiment_dir


@hydra.main(config_path="config/", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # setup distributed training
    is_ddp, world_size, device_map = utils.ddp.set_up_ddp()

    # validate configuration
    validate_cfg(cfg, is_ddp, world_size)

    # set directories
    model_dir = os.path.join(cluster.artifact_dir, "models")
    tokenizer_dir = os.path.join(cluster.artifact_dir, "tokenizers")

    # setup logging
    run, run_name, experiment_dir = setup_logging(cfg)

    # set seed
    utils.seed_everything(cfg.run.seed)

    # tokenizer
    tokenizer = get_tokenizer(cfg, tokenizer_dir)

    # data
    train_dataset, val_dataset, _ = get_data(cfg, tokenizer)

    # get model
    model = get_model(cfg, tokenizer, model_dir, device_map)

    # data collator
    if cfg.training.task == "seq2seq":
        data_collator = DataCollatorForSeq2SeqTimed(tokenizer, model=model, padding=True)
    elif cfg.training.task == "causal":
        data_collator = DataCollatorForLanguageModelingTimed(tokenizer, mlm=False)
    else:
        raise ValueError(f"Unknown task {cfg.training.task}")

    # Callbacks

    # generate text once for the callbacks and cache it. The other callbacks will just process the cached text.
    text_generator = utils.callbacks.CachedTextGenerator(
        tokenizer,
        model,
        train_dataset,
        val_dataset,
        data_collator,
        num_samples=cfg.callbacks.generate_text.num_samples,
        max_length=cfg.model.generation_max_length,
        batch_size=cfg.callbacks.generate_text.batch_size,
        prompt_length=cfg.callbacks.generate_text.prompt_length,
    )

    callbacks = []
    if cfg.callbacks.print.enabled:
        callbacks.append(
            utils.callbacks.PrintExamplesCallback(
                run,
                tokenizer,
                text_generator=text_generator,
                num_samples=cfg.callbacks.print.num_samples,
                to_console=cfg.callbacks.print.output_to_console,
                to_wandb=cfg.callbacks.print.output_to_wandb,
            )
        )
    if cfg.callbacks.seq2seq_metrics.enabled:
        callbacks.append(
            utils.callbacks.Seq2SeqMetricsCallback(
                run,
                tokenizer,
                val_dataset.name,
                train_dataset,
                val_dataset,
                model,
                MAX_LENGTH,
                num_examples=cfg.callbacks.seq2seq_metrics.num_examples,
            )
        )
    if cfg.callbacks.multi_dataset_seq2seq_metrics.enabled:
        assert cfg.data.name == "task_data_collection"
        callbacks.extend(
            [
                utils.callbacks.Seq2SeqMetricsCallback(
                    run, tokenizer, dsv.name, dst, dsv, model, MAX_LENGTH, num_examples=cfg.callbacks.multi_dataset_seq2seq_metrics.num_examples
                )
                for dst, dsv in zip(train_dataset.datasets, val_dataset.datasets)
            ]
        )
    if cfg.callbacks.metrics.enabled:
        callbacks.append(
            utils.callbacks.TextGenerationMetrics(
                run,
                tokenizer,
                text_generator=text_generator,
                compute_rouge=cfg.callbacks.metrics.rouge.enabled,
                compute_bleu=cfg.callbacks.metrics.bleu.enabled,
                compute_bertscore=cfg.callbacks.metrics.bertscore.enabled,
            )
        )

    callbacks.append(utils.callbacks.TimerCallback())

    # LoRA
    if cfg.training.lora.enabled:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            inference_mode=False,
            r=cfg.training.lora.rank,
            lora_alpha=cfg.training.lora.alpha,
            lora_dropout=cfg.training.lora.dropout,
        )
        model = get_peft_model(model, peft_config)
        print("Trainable Parameters:")
        model.print_trainable_parameters()

    # Trainer
    args = TrainingArguments(
        output_dir=experiment_dir,
        per_device_train_batch_size=cfg.optim.per_device_batch_size,
        per_device_eval_batch_size=cfg.optim.per_device_batch_size,
        eval_strategy="steps",
        eval_steps=cfg.log.eval_steps,
        logging_steps=cfg.log.logging_steps,
        gradient_accumulation_steps=cfg.optim.overall_batch_size // (cfg.optim.per_device_batch_size * world_size),
        num_train_epochs=cfg.optim.num_epochs,
        weight_decay=cfg.optim.weight_decay,
        warmup_steps=cfg.optim.warmup_steps,
        lr_scheduler_type=cfg.optim.lr_scheduler_type,
        learning_rate=cfg.optim.lr,
        save_steps=cfg.log.save_steps,
        push_to_hub=False,
        do_eval=True,
        torch_compile=cfg.run.compile,
        report_to="wandb",
        dataloader_num_workers=int(cluster.num_workers / torch.cuda.device_count()) if cfg.run.load_data_in_parallel else 0,
        eval_accumulation_steps=cfg.run.eval_accumulation_steps,
        group_by_length=cfg.training.group_by_length,
        ddp_find_unused_parameters=cfg.training.ddp_find_unused_parameters,
        **utils.get_mixed_precision(cfg.training.mixed_precision),
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
        compute_metrics=utils.metrics.get_metrics(cfg, tokenizer),
    )

    # workaround for ddp: set output dir on all ranks. The trainer requires this, but it cannot be communicated until the process group is initialised.
    run_name, experiment_dir = utils.ddp.broadcast_str(run_name, experiment_dir, source=0)
    utils.ddp.pprint_all_rank(f"Check that exchange of information worked: {run_name=}, {experiment_dir=}")
    trainer.args.output_dir = experiment_dir

    # evalute model before training
    if cfg.run.eval_before_training:
        with utils.disable_adapter_layers(model, cfg.training.lora.enabled):
            trainer.evaluate()

    # training
    trainer.train()
    trainer.save_model(experiment_dir)

    wandb.finish()


if __name__ == "__main__":
    try:
        utils.pprint(f"USING {torch.cuda.device_count()} DEVICES: [{os.environ['CUDA_VISIBLE_DEVICES']}]")
    except KeyError:
        utils.pprint(f"USING {torch.cuda.device_count()} DEVICES: ALL]")
    main()
    utils.pprint("Done!")
