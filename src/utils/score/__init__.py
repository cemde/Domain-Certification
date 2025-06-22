from typing import Dict, Any
from .task_data import _score_task_data, _aggregate_scores_task_data


def score_item(dataset: str, *args, **kwargs) -> Dict[str, Any]:
    if dataset == "task_data":
        return _score_task_data(*args, **kwargs)
    else:
        raise NotImplementedError()


def aggregate_scores(dataset: str, *args, **kwargs) -> Dict[str, float]:
    if dataset == "task_data":
        return _aggregate_scores_task_data(*args, **kwargs)


__all__ = [score_item, aggregate_scores]
