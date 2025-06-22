from enum import Enum
import os
from typing import Dict, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

import utils


class JudgeDecision(Enum):
    IN = "IN"  # In-distribution
    OUT = "OUT"  # Out-of-distribution, but natural language.
    INVALID = "INVALID"  # Out-of-distribution, not coherent.


@dataclass
class JudgeOutput:
    decision: JudgeDecision
    log_likelihoods: Dict[JudgeDecision, float]
    first_token_log_prob: Dict[JudgeDecision, float]

    def __dict__(self):
        return {
            "decision": self.decision.value,
            "log_likelihoods": {k.value: v for k, v in self.log_likelihoods.items()},
            "first_token_log_prob": {k.value: v for k, v in self.first_token_log_prob.items()},
        }

    def __iter__(self):
        return iter(self.__dict__())

    def to_dict(self):
        return self.__dict__()


class DecisionRegression(nn.Module):
    def __init__(self, model_path: str, decision: Enum):
        super().__init__()
        self.model_path = model_path
        self.model = nn.Linear(3, 3)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self._decision_enum = decision
        self.decision_map = {0: self._decision_enum.IN, 1: self._decision_enum.OUT, 2: self._decision_enum.INVALID}

    def __call__(self, x):
        xt = torch.tensor([x[self._decision_enum.IN], x[self._decision_enum.OUT], x[self._decision_enum.INVALID]]).float()
        model_output = self.model(xt).argmax().item()
        decision = self.decision_map[model_output]
        return decision


class Judge:
    DECISION = JudgeDecision

    def __init__(
        self,
        model_name_or_path,
        template: str,
        device_map: str,
        precision: str,
        cluster: utils.cluster.ClusterManager,
        log_template: bool = True,
        decision_strategy: str = "regression",
        regression_model_path: str = None,
    ):
        self.strategy = decision_strategy

        if decision_strategy == "regression":
            assert regression_model_path is not None, f"Regression model path needs to be provided for strategy {decision_strategy}"

        # load judge template
        template_path = f"utils/templates/ood_judge/{template}.txt"
        self.load_template(template_path)
        if log_template:
            self.log_template_artifact(template_path)

        # load judge model
        self.cache_dir = os.path.join(cluster.artifact_dir, "models")
        if precision in ["None", "none", None]:
            raise ValueError("Judge dtype needs to be explicitly set")
        j_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}[precision]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map=device_map, cache_dir=self.cache_dir, torch_dtype=j_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=self.cache_dir)

        if self.strategy == "regression":
            self.decision_regression = DecisionRegression(os.path.join(cluster.artifact_dir, "models", regression_model_path), self.DECISION)

    def load_template(self, template_path: str):
        with open(template_path, "r") as f:
            self.template = f.read()

        # the template consists of a list of options, followed by a separator, followed by the template

        # split the template by the separator
        template_list = self.template.split("\n")
        separator = "------"
        separator_idx = template_list.index(separator)
        self.template = self.template.split(separator)[1]

        # get the options. split each option by "::: "
        option_list = template_list[:separator_idx]
        decision_map = {option.split("::: ")[0].strip(): option.split("::: ")[1].strip() for option in option_list}
        self.decision_map = {
            decision_map["DECISION.IN"]: self.DECISION.IN,
            decision_map["DECISION.OUT"]: self.DECISION.OUT,
            decision_map["DECISION.INVALID"]: self.DECISION.INVALID,
        }
        self.decision_map_rev = {v: k for k, v in self.decision_map.items()}
        self.template = (
            self.template.replace("{{DECISION.IN}}", self.decision_map_rev[self.DECISION.IN])
            .replace("{{DECISION.OUT}}", self.decision_map_rev[self.DECISION.OUT])
            .replace("{{DECISION.INVALID}}", self.decision_map_rev[self.DECISION.INVALID])
        )

    def log_template_artifact(self, template_path: str):
        # save judge template as artifact
        artifact = wandb.Artifact(name="judge_template", type="template")
        artifact.add_file(local_path=template_path, name="txt_file")
        artifact.save()

    def __call__(self, sentence: str, topic: str) -> DECISION:
        filled_prompt = self.template.replace("{{sentence}}", sentence).replace("{{topic}}", topic)

        ll_id, prob_id = self.calculate_log_likelihood(filled_prompt, self.decision_map_rev[self.DECISION.IN])
        ll_od, prob_od = self.calculate_log_likelihood(filled_prompt, self.decision_map_rev[self.DECISION.OUT])
        ll_iv, prob_iv = self.calculate_log_likelihood(filled_prompt, self.decision_map_rev[self.DECISION.INVALID])

        # outputs = self.model(filled_prompt, max_new_tokens=5, do_sample=False)
        # response = outputs[0]["generated_text"][len(filled_prompt) :].strip()[:1]
        # decision = self.decision_map.get(response, self.DECISION.NA)

        ll_all = {self.DECISION.IN: ll_id, self.DECISION.OUT: ll_od, self.DECISION.INVALID: ll_iv}
        prob_all = {self.DECISION.IN: prob_id, self.DECISION.OUT: prob_od, self.DECISION.INVALID: prob_iv}

        decision = self.decide(ll_all, prob_all)

        return JudgeOutput(
            decision=decision,
            log_likelihoods=ll_all,
            first_token_log_prob=prob_all,
        )

    def calculate_log_likelihood(self, text: str, answer_text: str) -> Tuple[float, float]:
        text = text.replace("{{DECISION}}", answer_text)
        answer_id = self.tokenizer.encode(f" {answer_text}", add_special_tokens=False)[-1]
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        assert (
            inputs["input_ids"][0][-1].item() == answer_id
        ), f"Answer ID should be the last token in the sequence. Got {inputs['input_ids'][0][-1].item()} instead of {answer_id}"
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        log_likelihood = -outputs.loss.item()

        answer_log_prob = torch.log_softmax(outputs.logits[0, -1], dim=0)[answer_id].item()  # maybe remove log_softmax

        return log_likelihood, answer_log_prob

    def decide(self, log_likelihoods: Dict[DECISION, float], first_token_log_prob: Dict[DECISION, float]) -> DECISION:
        if self.strategy == "max_likelihood":
            decision = max(log_likelihoods, key=log_likelihoods.get)
        elif self.strategy == "max_prob_first_token":
            decision = max(first_token_log_prob, key=first_token_log_prob.get)
        elif self.strategy == "regression":
            decision = self.decision_regression(first_token_log_prob)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        return decision
