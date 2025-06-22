import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from .process_logits import postprocess_logits
from .inference import token_log_likelihood


@dataclass
class GenerateCertifiedOutput:
    """All logs are in base 2."""

    abstained: bool = None
    sequences: List[torch.LongTensor] = None
    d: float = None
    max_samples: int = None  # T or max_try
    num_samples: int = None
    log_probs: torch.FloatTensor = None
    log_probs_g: torch.FloatTensor = None
    log_ratios: List[float] = None
    token_likelihoods: torch.FloatTensor = None
    token_likelihoods_g: torch.FloatTensor = None
    upper_bounds: List[float] = None
    log_upper_bounds: List[float] = None
    N_token: List[int] = None
    log_ratios_normalized: List[float] = None


class MetaModel(nn.Module):
    log = math.log2
    EPS = 1e-12

    def __init__(
        self,
        tokenizer_model: AutoTokenizer,
        tokenizer_generator: AutoTokenizer,
        model: nn.Module,
        model_generation_config,
        generator: nn.Module,
        generator_config,
        k: float,
        T: int = 1,
        distribution_model: str = "y|x",
        distribution_generator: str = "y|x",
        divergence: str = "renyi_inf",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer_model = tokenizer_model
        self.tokenizer_generator = tokenizer_generator
        self.model = model
        self.model_generation_config = model_generation_config
        self.generator = generator
        self.generator_config = generator_config
        self.k = k
        self.T = T
        self.distribution_model = distribution_model
        self.distribution_generator = distribution_generator
        self.distribution_pair = f"F({distribution_model})||G({distribution_generator})"

        possible_divergence_pairs = ["F(y|x)||G(y|x)", "F(x+y)||G(x+y)", "F(y|x)||G(y)"]
        if self.distribution_pair not in possible_divergence_pairs:
            raise ValueError(f"Invalid distribution pair: {self.distribution_pair}. Required: {possible_divergence_pairs}")
        if divergence not in ["renyi_inf"]:
            raise ValueError(f"Unknown divergence: {divergence}")
        self.divergence = divergence

    def token_log_likelihood(self, tokenizer: AutoTokenizer, token_ids: torch.Tensor, logits: torch.Tensor):
        return token_log_likelihood(tokenizer, token_ids, logits, mask_eos=True, mask_padding=True)

    def valid_reject(self, x: torch.Tensor):
        """Performs our VALID rejection sampling algorithm."""

        assert x.dim() == 2, f"Input should be 2D, got {x.dim()}. batch x tokens"
        assert x.shape[0] == 1, "Only one input sequence is supported."

        # save history
        samples = []  # sentences drawn from f
        ds = []  # distance between f and g
        log_upper_bounds = []  # log2 upper bounds
        upper_bounds = []  # upper bounds
        log_probs_f_list = []
        log_probs_g_list = []
        log_ratios = []
        token_log_likelihoods_f_list = []
        token_log_likelihoods_g_list = []
        N_token_list = []

        i = 0
        sample_accepted = False

        while not sample_accepted and i < self.T:
            seq, token_log_likelihoods_f, log_probs_f = self._model_generate_with_likelihood(x)
            # s is entire sequence, log_probs_f and token_log_likelihoods_f are for the sequence to be certified.
            log_probs_g, token_log_likelihoods_g = self._generator_get_likelihood(seq, x)

            # log_probs: batch x N_certify x vocab (full distribution)
            # token_log_likelihood batch x N_certify

            assert log_probs_f.shape == log_probs_g.shape, "Log probabilities should have the same shape."
            assert token_log_likelihoods_f.shape == token_log_likelihoods_g.shape, "Token log likelihoods should have the same shape."
            assert token_log_likelihoods_f.dim() == token_log_likelihoods_g.dim() == 2, "Token log likelihoods should be 2D."

            # math exp has base e bcz the logits are base e. The log2 is for the cpk algorithm.
            log_prob_f = (token_log_likelihoods_f.sum() / math.log(2)).item()
            log_prob_g = (token_log_likelihoods_g.sum() / math.log(2)).item()

            # check if last token has 0 log likelihood, it is a masked EOS token, so we should not count it.
            N_token = token_log_likelihoods_f.shape[1] - int(
                token_log_likelihoods_f[0, -1].cpu() == token_log_likelihoods_g[0, -1].cpu() == 0.0
            )

            log_ratio = log_prob_f - log_prob_g
            log_upper_bound = self.k * N_token + math.log2(self.T) + log_prob_g
            upper_bound = 2**log_upper_bound
            if log_ratio <= self.k * N_token:
                sample_accepted = True

            # For debugging: uncomment below to print the log likelihoods and log ratios.
            # self._print_for_debugging(
            #     seq, x, i, token_log_likelihoods_f, token_log_likelihoods_g, log_prob_f, log_prob_g, log_ratio, log_upper_bound, N_token
            # )

            samples.append(seq.cpu())
            # d = self.get_d(log_probs_f.to(log_probs_g.device), log_probs_g)
            # ds.append(d)
            log_ratios.append(log_ratio)
            log_probs_f_list.append(log_probs_f)
            log_probs_g_list.append(log_probs_g)
            token_log_likelihoods_f_list.append(token_log_likelihoods_f.cpu())
            token_log_likelihoods_g_list.append(token_log_likelihoods_g.cpu())
            log_upper_bounds.append(log_upper_bound)
            upper_bounds.append(upper_bound)
            N_token_list.append(N_token)
            i += 1

        return_dict = {
            "abstained": not sample_accepted,
            "sequences": samples,
            "d": ds,
            "max_samples": self.T,
            "num_samples": i,
            "log_probs": log_prob_f,
            "log_probs_g": log_prob_g,
            "log_ratios": log_ratios,
            "token_likelihoods": token_log_likelihoods_f,
            "token_likelihoods_g": token_log_likelihoods_g,
            "log_upper_bounds": log_upper_bounds,
            "upper_bounds": upper_bounds,
            "N_token": N_token_list,
            "log_ratios_normalized": [(r / N if N > 0 else float("nan")) for r, N in zip(log_ratios, N_token_list)],
        }
        return return_dict

    def generate(self, x: torch.Tensor) -> GenerateCertifiedOutput:
        output_dict = self.valid_reject(x)
        return GenerateCertifiedOutput(**output_dict)

    def _model_generate_with_likelihood(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generates a response sequence from the model and calculates the log likelihood of the generated sequence.

        Args:
            x (torch.Tensor): a 2D tensor of token ids. batch x tokens

        Returns:
            sequences (torch.Tensor): a 2D tensor of token ids. batch x tokens. The prompt + response sequence.
            token_likelihoods (torch.Tensor): a 2D tensor of log probabilities of `sequences`. batch x tokens.
            log_probs (torch.Tensor): a 3D tensor of log probabilities. batch x tokens x vocab.
        """
        x = x.to(self.model.device)

        # generate sequences
        outputs = self.model.generate(
            input_ids=x,
            generation_config=self.model_generation_config,
        )

        sequences = outputs.sequences
        prompt_length = x.shape[1]
        responses = outputs.sequences[:, prompt_length:]

        # x == prompt; sequences == prompt + response; responses = response
        assert len(outputs.scores) == len(responses[0]), "Number of scores (post-processed logits) should equal length of generated output."

        if self.distribution_pair in ["F(y|x)||G(y)", "F(y|x)||G(y|x)"]:
            with torch.inference_mode():
                outputs_second_forward = self.model(sequences, labels=sequences)

            # logits = torch.stack(outputs.scores, dim=1)
            second_forward_logits = outputs_second_forward.logits[:, prompt_length - 1 : -1]
            model_top_k = self.model_generation_config.top_k if self.model_generation_config.top_k is not None else -1
            logits = self._postprocess_logits(second_forward_logits, temperature=self.model_generation_config.temperature, top_k=model_top_k)
            seq = responses
        elif self.distribution_pair == "F(x+y)||G(x+y)":
            with torch.inference_mode():
                outputs = self.model(sequences, labels=sequences)
            # the generation config for y is applied to x here. This is a design choice, not a necessity.
            logits = self._postprocess_logits(
                outputs.logits, temperature=self.model_generation_config.temperature, top_k=self.model_generation_config.top_k
            )
            seq = sequences

        assert seq.shape[:2] == logits.shape[:2], "Batch size and Number of scores (post-processed logits) and sequence should match."

        # keep seq as the sequence to certify and logits (post-processed) to calculate log probs.

        token_likelihoods, log_probs = self.token_log_likelihood(self.tokenizer_model, seq, logits)
        assert torch.isclose(torch.exp(log_probs).sum(-1), torch.tensor(1.0)).all(), "Expected log probabilities."
        # N = len(entire sequence) = prompt_length + response_length
        # N_Certify: length of sequence to certify. Either N or response_length
        # sequences shape: batch x N
        # token_likelihoods shape: batch x (N_Certify-1)
        # log_probs shape: batch x (N_Certify-1) x vocab
        assert token_likelihoods.shape == log_probs.shape[:2], "Token likelihoods and log probs should have the same shape."
        assert token_likelihoods.dim() == 2, "Token likelihoods should be 2D."
        assert log_probs.dim() == 3, "Log probs should be 3D."
        return sequences, token_likelihoods, log_probs

    @staticmethod
    def _truncate_logits_from_model_call(s: torch.Tensor, x: torch.Tensor, logits: torch.Tensor):
        prompt_length = x.shape[1]
        response = s[:, prompt_length:]
        logits = logits[:, (prompt_length - 1) : -1]

        return response, logits

    def _postprocess_logits(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 50):
        return postprocess_logits(logits, temperature=temperature, top_k=top_k)

    def _generator_get_likelihood(self, s: torch.Tensor, x: torch.Tensor):
        """
        Get the log likelihood of the generated sequence from the generator model.

        Args:
            s (torch.Tensor): The sequence to certify.
            x (torch.Tensor): The prompt sequence.

        Returns:
            torch.Tensor: The log probabilities of the generated sequence.
            torch.Tensor: The log likelihood of the generated sequence.
        """
        s = s.to(self.generator.device)  # entire sequence
        x = x.to(self.generator.device)  # prompt sequence
        prompt_length = x.shape[1]
        r = s[:, prompt_length:]  # response sequence

        # if we are certifying y|x or x,y we need to pass the entire sequence to the generator.
        if self.distribution_pair in ["F(y|x)||G(y|x)", "F(x+y)||G(x+y)"]:
            target = s
        # if we are certifying y, we only need to pass the response sequence to the generator.
        elif self.distribution_pair == "F(y|x)||G(y)":
            target = r

        # prepend target to condition on BOS
        bos = torch.tensor([[self.tokenizer_generator.bos_token_id]]).to(target.device)
        target = torch.cat([bos, target], dim=-1)

        with torch.inference_mode():
            outputs = self.generator(target, labels=target)

        # remove bos
        target = target[:, 1:]
        outputs.logits = outputs.logits[:, :-1, :]

        assert outputs.logits.shape[:2] == (1, target.shape[1]), "Batch size and sequence length should match."

        # if y|x, the outputs are the logits of the entire sequence. We need to truncate the logits to get the logits of the response.
        if self.distribution_pair in ["F(y|x)||G(y|x)"]:
            assert False
        elif self.distribution_pair == "F(y|x)||G(y)":
            logits = outputs.logits
            seq = r
        elif self.distribution_pair == "F(x+y)||G(x+y)":
            assert False

        logits = self._postprocess_logits(logits, temperature=self.generator_config.temperature, top_k=self.generator_config.top_k)
        token_likelihoods, log_probs = self.token_log_likelihood(self.tokenizer_generator, seq, logits)
        # N = len(entire sequence) = prompt_length + response_length
        # N_Certify: length of sequence to certify. Either N or response_length
        # token_likelihoods shape: batch x (N_Certify-1)
        # log_probs shape: batch x (N_Certify-1) x vocab
        return log_probs, token_likelihoods

    def _print_for_debugging(
        self, seq, x, i, token_log_likelihoods_f, token_log_likelihoods_g, log_prob_f, log_prob_g, log_ratio, log_upper_bound, N_token
    ):
        """Prints the log likelihoods and log ratios for debugging purposes."""
        seq_text = self.tokenizer_model.decode(seq[0])
        prompt_text = self.tokenizer_model.decode(x[0], skip_special_tokens=True)
        response_text = self.tokenizer_model.decode(seq[0, x.shape[1] :], skip_special_tokens=True)
        print("")
        print("")
        print("=" * 80)
        print(f"Sample {i}:")
        print("=" * 80)
        print("Seq[0]")
        print(seq[0])
        print("Response:")
        print(seq[0][x.shape[1] :])
        print("Prompt:")
        print(prompt_text)
        print(repr(prompt_text))
        print("=" * 80)
        print("Response:")
        print(response_text)
        print(repr(response_text))
        print("=" * 80)
        print(f"k={self.k}, T={self.T}")
        print("N_token:", N_token)
        print(f"Number of tokens Model:     {token_log_likelihoods_f.shape}")
        print(f"Number of tokens Generator: {token_log_likelihoods_g.shape}")
        print(f"Ln Prob F:  {token_log_likelihoods_f.sum()}")
        print(f"Ln Prob G:  {token_log_likelihoods_g.sum()}")
        print("Log2 Prob F:", log_prob_f)
        print("Log2 Prob G:", log_prob_g)
        print("Log2 Ratio:", log_ratio)
        print("Log2 Upper Bound:", log_upper_bound)
        print(f"KN={self.k * N_token}")
