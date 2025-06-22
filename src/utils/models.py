import torch
import utils

GPT_SIZES = {
    "gpt-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 24-layer, 1024-hidden, 16-heads, 345M parameters.
    "gpt": dict(n_layer=12, n_head=12, n_embd=768),
    "gpt-xs": dict(n_layer=6, n_head=6, n_embd=384),
    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
    "gpt-pico": dict(n_layer=2, n_head=2, n_embd=24),
}


def set_model_precision(model, precision: str):
    if precision == "fp16":
        model = model.to(dtype=torch.float16)
    elif precision == "fp32":
        model = model.to(dtype=torch.float32)
    elif precision == "bf16":
        if not utils.supports_bfloat16():
            raise ValueError("bfloat16 not supported on this device.")
        model = model.to(dtype=torch.bfloat16)
    elif precision in ["None", "none", None]:
        pass
    else:
        raise ValueError(f"Unknown precision {precision}")

    return model
