from typing import Tuple
from transformers import AutoTokenizer
from omegaconf import DictConfig
import os


def get_single_tokenizer(net_cfg: DictConfig, base_dir: str) -> AutoTokenizer:
    tokenizer_name_or_path = net_cfg.tokenizer.name_or_path if net_cfg.tokenizer.name_or_path is not None else net_cfg.name_or_path
    tokenizer_source = net_cfg.tokenizer.source if net_cfg.tokenizer.name_or_path is not None else net_cfg.source
    if tokenizer_source == "hf":
        pass
    elif tokenizer_source == "local":
        tokenizer_name_or_path = os.path.join(base_dir, tokenizer_name_or_path)  # the tokenizer should be saved with the model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=base_dir, padding_side="right")
    if net_cfg.tokenizer.add_pad_token:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer, tokenizer_name_or_path, tokenizer_source


def get_tokenizers(cfg: DictConfig, base_dir: str) -> Tuple[AutoTokenizer, AutoTokenizer]:
    """Expects config to have cfg.model.tokenizer and cfg.generator.tokenizer namespace."""
    # model tokenizer
    tokenizer_model, tokenizer_model_name_or_path, tokenizer_model_source = get_single_tokenizer(cfg.model, base_dir)
    print(f"model tokenizer = {tokenizer_model_name_or_path} | Source: ({tokenizer_model_source}). Vocab size: {tokenizer_model.vocab_size}")

    # generator tokenizer
    tokenizer_generator, tokenizer_generator_name_or_path, tokenizer_generator_source = get_single_tokenizer(cfg.generator, base_dir)
    print(
        f"generator tokenizer = {tokenizer_generator_name_or_path} | Source: ({tokenizer_generator_source}). Vocab size: {tokenizer_generator.vocab_size}"
    )

    # extremely unrobust sanity check to catch the biggest errors
    if cfg.tokenizers_match:
        test_text = "This is a test. A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
        assert tokenizer_model.encode(test_text) == tokenizer_generator.encode(test_text)
        assert tokenizer_model.vocab_size == tokenizer_generator.vocab_size

    return tokenizer_model, tokenizer_generator
