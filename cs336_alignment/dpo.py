"""
CS336 Assignment 5 - DPO (Direct Preference Optimization) Implementation.

Implements:
- compute_per_instance_dpo_loss
"""

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase


# Alpaca SFT template
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)


def _get_sequence_log_prob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute total log probability of response tokens under the model."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    response_mask = response_mask.to(device)

    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids).logits

    log_probs = F.log_softmax(logits, dim=-1)
    per_token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    # Sum log probs over response tokens only
    return (per_token_log_probs * response_mask).sum(dim=-1)


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """Compute per-instance DPO loss.
    
    loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    where log_ratio = log_pi_theta(y|x) - log_pi_ref(y|x)
    """
    eos_token = tokenizer.eos_token if tokenizer.eos_token else ""

    # Format with Alpaca template and append EOS
    chosen_text = ALPACA_TEMPLATE.format(instruction=prompt, response=response_chosen) + eos_token
    rejected_text = ALPACA_TEMPLATE.format(instruction=prompt, response=response_rejected) + eos_token

    # Tokenize prompt separately to know where response starts
    prompt_text = ALPACA_TEMPLATE.format(instruction=prompt, response="")
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    # Tokenize full sequences
    chosen_ids = tokenizer.encode(chosen_text, add_special_tokens=False)
    rejected_ids = tokenizer.encode(rejected_text, add_special_tokens=False)

    # Create input_ids and labels (shifted by 1)
    def make_tensors(ids):
        seq_len = len(ids) - 1
        input_ids = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0)
        # Response mask: 1 for response tokens in labels
        # Label at position j corresponds to token ids[j+1]
        # Response starts at position prompt_len in the original ids
        # So in labels, response starts at position prompt_len - 1
        response_mask = torch.zeros(1, seq_len, dtype=torch.float32)
        resp_start = prompt_len - 1
        response_mask[0, resp_start:] = 1.0
        return input_ids, labels, response_mask

    chosen_input_ids, chosen_labels, chosen_mask = make_tensors(chosen_ids)
    rejected_input_ids, rejected_labels, rejected_mask = make_tensors(rejected_ids)

    # Compute log probs under policy model
    lm.eval()
    lm_ref.eval()

    with torch.no_grad():
        log_prob_chosen_policy = _get_sequence_log_prob(lm, chosen_input_ids, chosen_labels, chosen_mask)
        log_prob_rejected_policy = _get_sequence_log_prob(lm, rejected_input_ids, rejected_labels, rejected_mask)
        log_prob_chosen_ref = _get_sequence_log_prob(lm_ref, chosen_input_ids, chosen_labels, chosen_mask)
        log_prob_rejected_ref = _get_sequence_log_prob(lm_ref, rejected_input_ids, rejected_labels, rejected_mask)

    # DPO loss
    log_ratio_chosen = log_prob_chosen_policy - log_prob_chosen_ref
    log_ratio_rejected = log_prob_rejected_policy - log_prob_rejected_ref

    loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    return loss.squeeze()
