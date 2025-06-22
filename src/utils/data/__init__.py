# ruff: noqa: F401

from omegaconf import DictConfig

from torch.utils.data import Dataset

# from .addition import AdditionDataset
from .sorting_data import (
    TASKS,
    TaskCollectionDataset,
    TaskDataset,
    TaskDatasetPrompting,
    TaskDatasetTraining,
    generate_sequences,
    print_examples,
)

from .shakespeare import TinyShakespeareDataset
from .pubmedqa import PubMedQADataset, PubMedQAWithGeneratedResponsesDataset, PubMedQAScoringDataset, PUBMEDQA_SCORING_MARKER
from .squad import SQuADDataset, SQuADWithGeneratedResponsesDataset
from .mmlu import MMLUQADataset, MMLUScoringDataset, mmlu_categories, MMLU_SCORING_MARKER


def get_dataset_config_name(cfg: DictConfig) -> str:
    """Get the dataset config name for the given dataset configuration.

    Args:
        cfg: The dataset configuration. Assumes that cfg.data namespace exists with cfg.data.name as the dataset name and other properties as defined by dataset class.

    Returns:
        The dataset config name for the given dataset configuration. This is a string identified for the dataset configuration regarding its domain (e.g. indicating SQuAD with only medical questions).

    """
    if cfg.data.name == "task_data":
        return TaskDatasetTraining.get_dataset_config_name(cfg)
    elif cfg.data.name == "pubmedqa":
        return PubMedQADataset.get_dataset_config_name(cfg)
    elif cfg.data.name == "squad":
        return SQuADDataset.get_dataset_config_name(cfg)
    elif cfg.data.name == "pubmedqa_generated":
        return PubMedQAWithGeneratedResponsesDataset.get_dataset_config_name(cfg)
    elif cfg.data.name == "squad_generated":
        return SQuADWithGeneratedResponsesDataset.get_dataset_config_name(cfg)
    elif cfg.data.name == "mmlu":
        return MMLUQADataset.get_dataset_config_name(cfg)
    else:
        raise ValueError(f"Unknown dataset {cfg.data.name}")
