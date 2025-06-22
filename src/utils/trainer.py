import torch
from transformers import Trainer


class DebugTrainer(Trainer):
    def training_step(self, model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        # Get distributed context
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        # First training step inspection
        if self.state.global_step == 0:
            input_ids = inputs.get("input_ids")
            for i in range(3):
                if input_ids is not None:
                    samples = input_ids[i].cpu().numpy()
                    print(f"[RANK: {rank}] Input IDs sample {i}: {samples[:15]}")
                else:
                    print(f"[RANK: {rank}] No input_ids found in batch")

        return super().training_step(model, inputs)
