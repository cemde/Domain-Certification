from typing import Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def token_log_likelihood(
    tokenizer: AutoTokenizer,
    token_ids: torch.Tensor,
    logits: torch.Tensor,
    mask_padding: bool = True,
    mask_eos: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate log likelihood of a given sentence of tokens.

    Args:
        tokenizer (AutoTokenizer): A HuggingFace tokenizer
        token_ids (torch.Tensor): a 2D tensor of token ids. batch x tokens
        logits (torch.Tensor): a 3D tensor of logits. batch x tokens x vocab.
        mask_padding (bool, optional): If true, the loglikelihood of padding tokens will be set to 0. Defaults to True.
        mask_eos (bool, optional): If true, the loglikelihood of EOS tokens will be set to 0. Defaults to True.

    Returns:
        log_likelihoods (torch.Tensor): a 2D tensor of log probabilities of `token_ids`. batch x tokens.
        logsoftmax (torch.Tensor): a 3D tensor of log probabilities. batch x tokens x vocab.

    """
    # get log likelihood of give sentence for all tokens that are generated.

    assert token_ids.dim() == 2, f"sentence should be 2D, got {token_ids.dim()}. batch x tokens"
    assert logits.dim() == 3, f"logits should be 3D, got {logits.dim()}. batch x tokens x vocab"
    assert token_ids.shape[0] == logits.shape[0], "batch size mismatch"
    assert token_ids.shape[1] == logits.shape[1], "sequence length mismatch"

    # Get the log probabilities of the response_ids
    logsoftmax = F.log_softmax(logits, dim=-1)
    log_likelihoods = torch.gather(logsoftmax, 2, token_ids.unsqueeze(-1)).squeeze(-1)

    # mask out padding tokens
    if mask_padding:
        mask = token_ids.ne(tokenizer.pad_token_id)
        log_likelihoods = log_likelihoods * mask

    # remove EOS token
    if mask_eos:
        mask = mask & token_ids.ne(tokenizer.eos_token_id)
        log_likelihoods = log_likelihoods * mask

    return log_likelihoods, logsoftmax
