from typing import Callable
from transformers import AutoTokenizer
import numpy as np
from omegaconf import DictConfig


def compute_perplexity(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute perplexity from logits and labels.

    Args:
        logits (np.ndarray): Predicted logits, shape (batch_size, n_token, vocab_size)
        labels (np.ndarray): True labels, shape (batch_size, n_token)

    Returns:
        float: Perplexity score
    """
    # Step 1: Assert shape compatibility
    assert logits.ndim == 3, "Logits should be 3-dimensional"
    assert labels.ndim == 2, "Labels should be 2-dimensional"
    assert logits.shape[:2] == labels.shape, "Batch size and n_token dimensions should match"

    batch_size, n_token, vocab_size = logits.shape

    # Step 2: Convert logits to probabilities using softmax
    # To avoid numerical instability, use log-sum-exp trick
    max_logits = np.max(logits, axis=-1, keepdims=True)
    log_probs = logits - max_logits - np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))

    # Step 3: Gather log probabilities for the true labels
    log_probs_for_labels = np.take_along_axis(log_probs, labels[..., None], axis=-1).squeeze(-1)

    # Step 4: Compute the cross-entropy loss
    cross_entropy_loss = -np.mean(log_probs_for_labels)

    # Step 5: Compute perplexity
    perplexity = np.exp(cross_entropy_loss)

    return perplexity


def get_metrics(cfg: DictConfig, tokenizer: AutoTokenizer) -> Callable:
    # Load metrics
    perplexity = compute_perplexity if cfg.metrics.perplexity.enabled else None

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        # Decode predictions and labels
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        metrics = {}

        # Perplexity
        if perplexity is not None:
            metrics["perplexity"] = perplexity(logits, labels)

        return metrics

    return compute_metrics
