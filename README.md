# Domain Certification

[![Project-Homepage](https://img.shields.io/badge/Project-Webpage-db6fc0.svg)](https://cemde.github.io/Domain-Certification-Website/)
[![arXiv](https://img.shields.io/badge/arXiv-2502.19320-b31b1b.svg)](https://arxiv.org/abs/2502.19320)
[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-7ACA2C.svg)](https://iclr.cc/virtual/2025/poster/30364)
[![Huggingface](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-blue.svg)](https://huggingface.co/collections/cemde/domain-certification-67ba4fb663f8d1348c3c2263)

**Certify you Large Language Model (LLM)!**

With the code in this repository you can reproduce the workflows we use in our ICLR 2025 paper to achieve Domain Certification using our VALID algorithm.

## Getting Started

1. Install the Python environment. `conda env create -f environment.yaml`.
2. This repository uses the [`ClusterManager`](src/utils/cluster.py) class to configure paths and system variables automatically To use this, first set an environment variable (e.g. using `.env` or in your `~/.bashrc`) `CLUSTER_NAME=<SYSTEM_A_USER_123>` and then add a configuration under this name into [`src/config/system.yaml`](src/config/system.yaml). Once set, all paths in the scripts will be automatically augmented with the directories specified there.
3. This Repository uses logging using [Weights & Biases (W&B)](https://wandb.ai/site/). Create an account and follow the instructions to login locally. Alternatively, you can edit the code to use other logging such as [Neptune AI](https://neptune.ai) with minimal effort. Finally, search for `wandb.init` in the code base and configure to your expectations.
4. Run scripts as specified below. Each script uses [`Hydra`](https://hydra.cc) to configure the runs using `yaml` files in [`src/config/*.yaml`](src/config). This can be overwritten in the command line (see examples below).

## Entry Points

| Path                                                                                   | Description                                                                                                                             |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [`src/certified_inference.py`](src/certified_inference.py)                             | Performs the VALID algorithm for certified inference with up to $T$ steps. Can be used to create datasets with LLM-generated responses. |
| [`src/train_model.py`](src/train_model.py)                                             | Train a model. This can be full fine-tuning, LoRA fine-tuning or training a model from scratch.                                         |
| [`src/multiple_model_likelihood_dataset.py`](src/multiple_model_likelihood_dataset.py) | Loads two models and a datatset. Compares the likelihood of the samples in the dataset under both models. Saves to `pickle` file.       |
| [`src/test_model_benchmark.py`](src/test_model_benchmark.py)                           | ....                                                                                                                                    |
| [`src/create_task_dataset.py`](src/create_task_dataset.py)                             | Creates the TaskData Dataset (see Appendix). It is useful for debugging.                                                                |
| [`src/create_tokenizer.py`](src/create_tokenizer.py)                                   | Creates a Tokenizer once the TaskData dataset is generated.                                                                             |

## Notebooks

| Path                                      | Description                                                                                                                                                                                       |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/notebooks/results_likelihoods.ipynb` | Loads the JSON as saved by `src/multiple_model_likelihood_dataset.py` for ID and OOD datasets and creates figures and metrics.                                                                    |
| `src/notebooks/results_benchmark.ipynb`   | Loads the JSON as saved by `src/multiple_model_likelihood_dataset.py` and benchmark results as saved by `src/test_model_benchmark.py` and creates figures and metrics for certified benchmarking. |

## Workflow

### Generate Predictions as Dataset

We conduct experiments using both ground-truth datasets and LLM-generated predictions. To streamline the inference workflow, we provide multiple versions of each dataset:

- **Ground-truth version**: Contains original queries and their corresponding ground-truth responses

  - Example: `cfg.data=pubmedqa` loads the `PubMedQADataset` dataset

- **Generated version**: Contains original queries paired with LLM-generated responses
  - Example: `cfg.data=pubmedqa_generated` loads the `PubMedQAWithGeneratedResponsesDataset` dataset

The `*_generated` datasets include configuration options to load generated responses from a `json` file produced by `certified_inference.py`. For instance, with `pubmedqa_generated`, you can use the config parameter `cfg.generated_output_paths.test` to specify the response file location.

To generate these responses, use the following command:

```bash
python certified_inference.py model.temperature=1.0 generator.temperature=1.0 model.generation_max_new_tokens=256 meta_model.k=4 meta_model.T=1 data.split=test data=pubmedqa data.test_size=100
```

> **Note**: While it may seem counterintuitive to use `certified_inference.py` before operations like fine-tuning on `pubmedqa_generated`, this process is designed to only extract the generated responses from the output JSON file. You can ignore the other contents of the file for these purposes.

### Train a model

**Example 1: Train a model from scratch**

- on responses from PubMedQA
- with a batch size of 16 per GPU
- a total batch size of 128 (gradient accumulation will be configured based on this information)
- using a small GPT-2-like model architecture

The parameters are loaded from [`config/train.yaml`](src/config/train.yaml) and then partially overwritten with the command below. For example,

```bash
torchrun --nproc_per_node=2 train_model.py --config-name train data=pubmedqa data.return_sequence=partial-response tokenizer.name=meta-llama/Meta-Llama-3-8B tokenizer.source=hf tokenizer.add_pad_token=True optim.lr=0.01 optim.num_epochs=1 model.architecture=gpt-xs model.source=local
```

We use `torchrun` for distributed data-parallel training.

> **WARNING:** The `Dataset` classes use the huggingface typical cashing mechanism. On the first time you run them, they cache the tokenized data. If you haven't tokenized the dataset doing so within `torchrun` can lead to issues. Run a singular instance of the script first (`python train_model.py ...`) to tokenize, and then `torchrun ... train_model.py ...` will load the cached tokenized dataset. The same is true for loading models from huggingface.

**Example 2: Fine-tune a model**

Fine-tune a model from huggingface:

```bash
torchrun --nproc_per_node=2 train_model.py --config-name train data=pubmedqa_generated data.return_sequence=partial-response tokenizer.name=cemde/Domain-Certification-MedQA-Guide-Base tokenizer.source=hf tokenizer.add_pad_token=False optim.lr=0.01 optim.num_epochs=1 model.architecture=cemde/Domain-Certification-MedQA-Guide-Base model.source=hf
```

If you want to fine-tune a model from huggingface just use `tokenizer.source=hf` and `model.source=hf`, otherwise `<...>.source=local`.

If you want to use LoRA use the `training.lora` namespace in the config.

### Inference with LLM and Guide Model

To perform certification you have two options:

- **Option 1:** You can use the `multiple_model_likelihood_dataset.py` to certify on a defined dataset. The script loads model $L$ and model $G$ and a dataset. It obtains the likelihood of each sample in the dataset under both models and saves results as `pickle` file, so that we can investigate the likelihood ratios and downstream metrics.
- **Option 2:** You can use the `certified_inference.py` script as introduced above, which performs the VALID algorithm: It loads two models and a dataset and then uses the query from the dataset to sample from the large model $L$ up to $T$ times.

**Option 1:**

```bash
python multiple_model_likelihood_dataset.py inference.prompt_length=dataset tokenizers_match=False data=pubmedqa data.test_size=100
```

This runs the script with the models as defined in [`src/config/model_likelihood.yaml`](src/config/model_likelihood.yaml). This saves a `pickle` file with the likelihoods that can then be analysed with the [`src/notebooks/results_likelihoods.ipynb`](src/notebooks/results_likelihoods.ipynb) notebook.

**Option 2:**

```bash
python certified_inference.py model.temperature=1.0 generator.temperature=1.0 model.generation_max_new_tokens=256 meta_model.k=4 meta_model.T=1 data.split=test data=pubmedqa data.test_size=100
```

This runs the VALID algorithm using $T=1$ and $k=4$.

### Evaluate Certificate

To evaluate the certificates refer to Jupyter notebook [`src/notebooks/results_likelihoods.ipynb`](src/notebooks/results_likelihoods.ipynb). This loads the results generated by `multiple_model_likelihood_dataset.py` results for ID and OOD data. It will then generate the key figures in our paper.

In the second cell, you will find this:

```python
# SETUP THESE VARIABLES
run_names = ["6e43-10b0", "6385-5556"]

...

##### LOAD BY RUN NAMES
paths = [
    f"/path/to/your/artifact_dir/model_likelihood/{run_name}/model_likelihood.pkl" for run_name in run_names
]
```

In `paths` set the correct artifact directory you use elsewhere. Inside `run_names` only specify the random ID that `multiple_model_likelihood_dataset.py` generates. It should look the same as above.


### Benchmarking

To run benchmarking as proposed in our work, you need to run 2 scripts to obtain different results:

- **Path A - Benchmarking:** Perform multiple-choice benchmarking to determine which questions are answered correctly.
- **Path B - Certification:** Perform certification on the correct answer to determine whether it would be accepted by VALID.

**Path A - Benchmarking:** Use the following command to generate a pickle file with benchmark results:

```bash
python test_model_benchmark.py data=pubmedqa data.variant=scoring
```

**Path B - Certification:** Use this to get the model likelihoods:

```bash
python multiple_model_likelihood_dataset.py data=pubmedqa inference.prompt_length=dataset tokenizers_match=False data=pubmedqa data.test_size=100
```

These should then be loaded into [`src/notebooks/results_likelihoods.ipynb`](src/notebooks/results_likelihoods.ipynb), which saves a table with certificates.

**Combine**

Then finally, these results can be combined in the [`src/notebooks/results_benchmark.ipynb`](src/notebooks/results_benchmark.ipynb) notebook. In this notebook you specify the pickle and JSON files generated with the two commands above and calculate the certified benchmark results.

## Contributing

Contributions are very welcome. Perform formatting and linting with `ruff`. It will automatically adopt the settings specified in the [`pyproject.toml`](pyproject.toml) file. Finally open a PR and request a review from [@cemde](https://github.com/cemde).

## Citation

```bibtex
@inproceedings{
emde2025shh,
title={Shh, don't say that! Domain Certification in {LLM}s},
author={Cornelius Emde and Alasdair Paren and Preetham Arvind and Maxime Guillaume Kayser and Tom Rainforth and Bernard Ghanem and Thomas Lukasiewicz and Philip Torr and Adel Bibi},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://arxiv.org/abs/2502.19320}
}
```

---
