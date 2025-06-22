import torch


def postprocess_logits(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
    """
    Postprocess logits. This function applies temperature scaling and top-k filtering to logits.

    Args:
        logits (torch.Tensor): Logits to postprocess.
        temperature (float, optional): Temperature scaling. Defaults to 1.0.
        top_k (int, optional): Top-k filtering. Defaults to 50.

    Returns:
        torch.Tensor: Postprocessed logits.
    """
    # postprocess logits
    if temperature == 0:
        raise ZeroDivisionError("Temperature has to be strictly positive.")
    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        topk_values, topk_indices = torch.topk(logits, top_k, dim=-1)
        mask = torch.full_like(logits, False, dtype=torch.bool)
        mask.scatter_(-1, topk_indices, True)
        logits = logits.clone()  # logits is inference tensor and has to be cloned to allow for inplace operations
        logits[~mask] = float("-inf")
    return logits
