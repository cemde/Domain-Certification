import os
import subprocess

import utils
from tokenizers import Tokenizer, models, trainers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

cluster = utils.cluster.ClusterManager()


def count_lines_fast(filename: str) -> int:
    result = subprocess.run(["wc", "-l", filename], text=True, capture_output=True)
    if result.returncode > 0:
        raise Exception(f"Error counting lines: {result.stderr}")
    line_count = int(result.stdout.split()[0])
    return line_count


def train_bpe_tokenizer(dataset_path: str, vocab_size: int = 400) -> None:
    # Initialize a tokenizer with BPE model with start and end token
    tokenizer_base = Tokenizer(models.BPE())

    # Specify special tokens for the tokenizer
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]  # "<s>", "</s>"]
    tokenizer_base.add_special_tokens(special_tokens)
    tokenizer_base.pre_tokenizer = Whitespace()
    tokenizer_base.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )

    # Initialize the trainer with the desired vocabulary size
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

    # Get number of sequences in the dataset
    text_file_path = os.path.join(dataset_path, "all.txt")
    print(f"Generating tokenizer from {text_file_path}")
    print(f"File has {count_lines_fast(text_file_path):,} lines.")

    print("Query tokens: ")
    with open(query_token_path, "r") as f:
        for i, line in enumerate(f):
            print(line.strip(), end=" ")
    print("")

    # Train the tokenizer on the dataset
    tokenizer_base.train(files=[text_file_path, query_token_path], trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_base)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token = "<pad>"

    # # Save the tokenizer to the disk
    # tokenizer_path = os.path.join(dataset_path, "tokenizer")
    tokenizer_path = os.path.join(cluster.artifact_dir, "tokenizers", "TaskData")
    os.makedirs(tokenizer_path, exist_ok=True)

    # tokenizer.save(tokenizer_path)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer trained and saved to {tokenizer_path}")


if __name__ == "__main__":
    dataset_path = os.path.join(cluster.data_dir, "TaskData", "MOD_mixed_TASK_add_LEN_49_INT_99_CHAR_249_CSL_False")
    query_token_path = os.path.join(cluster.data_dir, "TaskData", "query_token.txt")
    train_bpe_tokenizer(dataset_path, 400)
