import argparse
import hashlib
import json
import os

import utils

cluster = utils.cluster.ClusterManager()

parser = argparse.ArgumentParser()
parser.add_argument("--dataset-modality", type=str, default="mixed")
parser.add_argument("--task", type=str, default="sort", choices=utils.data.TASKS)
parser.add_argument("--n-train", type=int, default=int(5e6))
parser.add_argument("--n-val", type=int, default=1024)
parser.add_argument("--n-test", type=int, default=8192)
parser.add_argument("--max-sequence-length", type=int, default=99)
parser.add_argument("--population-num-integers", type=int, default=99)
parser.add_argument("--population-num-chars", type=int, default=249)
parser.add_argument("--constant-sequence-length", action="store_true")
parser.add_argument("--seed", type=int, default=23633)
args = parser.parse_args()

utils.seed_everything(args.seed)


def main(
    dataset_modality: str,
    task: str,
    n_train: int,
    n_val: int,
    n_test: int,
    max_sequence_length: int,
    population_num_integers: int,
    population_num_chars: int,
    constant_sequence_length: bool,
    unqiue_sequences: bool = True,
    **kwargs,
):
    assert unqiue_sequences, "Only unique sequences are supported at the moment."
    n_total = n_train + n_val + n_test
    sequences = set()

    seq_lengths = [0]

    print(f"Generating sequences. Reapeating until {n_total} unique sequences are generated.")
    while len(sequences) < n_total:
        n_remaining = n_total - len(sequences)
        new_sequences = utils.data.generate_sequences(
            modality=dataset_modality,
            task=task,
            N=n_remaining,
            T=max_sequence_length,
            num_int=population_num_integers,
            num_char=population_num_chars,
            constant_length=constant_sequence_length,
        )
        sequences.update(new_sequences)
        seq_lengths.append(len(sequences))

    print(f"Generated {len(sequences)} unique sequences.")
    print(f"Sequence lengths: {seq_lengths}")

    sequences = list(sequences)

    train_data = sequences[:n_train]
    val_data = sequences[n_train : n_train + n_val]
    test_data = sequences[n_train + n_val :]

    config = dict(
        seed=args.seed,
        modality=dataset_modality,
        task=task,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        max_seq_length=max_sequence_length,
        num_int=population_num_integers,
        num_char=population_num_chars,
        constant_seq_length=constant_sequence_length,
    )

    config_train = utils.data.TaskDataset.Config(**config, n=n_train, split="train")
    config_val = utils.data.TaskDataset.Config(**config, n=n_val, split="val")
    config_test = utils.data.TaskDataset.Config(**config, n=n_test, split="test")
    config_all = utils.data.TaskDataset.Config(**config, n=n_total, split="all")

    train_data = {"config": config_train.model_dump(), "data": train_data}
    val_data = {"config": config_val.model_dump(), "data": val_data}
    test_data = {"config": config_test.model_dump(), "data": test_data}
    all_data = {"config": config_all.model_dump(), "data": sequences}

    data_dir = os.path.join(
        cluster.data_dir,
        "TaskData",
        utils.data.TaskDataset.get_disk_subpath(
            dataset_modality, task, max_sequence_length, population_num_integers, population_num_chars, constant_sequence_length
        ),
    )
    os.makedirs(data_dir, exist_ok=True)

    file_contents = {"train": train_data, "val": val_data, "test": test_data, "all": all_data}
    for file_name, file_data in file_contents.items():
        save_path = os.path.join(data_dir, f"{file_name}.json")
        with open(save_path, "w") as f:
            json.dump(file_data, f)
        print(f"Saved {len(file_data['data']):,} sequences for '{file_name}' to {save_path}.")

    # get md5 has per file and save it to "hash.txt"
    hash_file_path = os.path.join(data_dir, "hash.txt")
    with open(hash_file_path, "w") as f:
        for file_name in file_contents.keys():
            file_path = os.path.join(data_dir, f"{file_name}.json")
            with open(file_path, "rb") as f2:
                file_hash = hashlib.md5(f2.read()).hexdigest()
            f.write(f"{file_name}: {file_hash}\n")

    print(f"Saved hash to {data_dir}.")

    # save joined text
    all_path = os.path.join(data_dir, "all.txt")
    with open(all_path, "w") as f:
        for seq in sequences:
            f.write(f"{seq[0]} {seq[1]}\n")

    print(f"All coherent sequences saved to {all_path}.")


if __name__ == "__main__":
    # print args
    for arg in vars(args):
        print(f"{arg:<30}: {getattr(args, arg)}")
    main(**vars(args))
