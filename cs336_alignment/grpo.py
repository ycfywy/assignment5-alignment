"""
CS336 Assignment 5 - GRPO Implementation.

Implements all GRPO components:
- tokenize_prompt_and_output
- get_response_log_probs
- compute_rollout_rewards
- compute_group_normalized_rewards
- compute_policy_gradient_loss
- aggregate_loss_across_microbatch
- grpo_train_step
"""

from typing import Callable, Literal

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt and output, construct response mask."""
    batch_size = len(prompt_strs)

    # Tokenize each prompt and output separately without special tokens
    all_ids = []
    prompt_lens = []
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        combined = prompt_ids + output_ids
        all_ids.append(combined)
        prompt_lens.append(len(prompt_ids))

    # Find max combined length
    max_len = max(len(ids) for ids in all_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Sequence length for input_ids/labels is max_len - 1
    seq_len = max_len - 1

    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    response_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

    for i, (ids, p_len) in enumerate(zip(all_ids, prompt_lens)):
        n = len(ids)
        # input_ids[i] = ids[:seq_len] (first seq_len tokens, or all if shorter)
        fill_len = min(n, seq_len)
        input_ids[i, :fill_len] = torch.tensor(ids[:fill_len], dtype=torch.long)
        # labels[i] = ids[1:seq_len+1] (shifted by 1)
        label_fill_len = min(n - 1, seq_len)
        labels[i, :label_fill_len] = torch.tensor(ids[1:1 + label_fill_len], dtype=torch.long)
        # response_mask: aligned with labels
        # labels[j] = ids[j+1]. ids[j+1] is in response if j+1 >= p_len, i.e., j >= p_len-1
        resp_start = p_len - 1
        resp_end = label_fill_len
        if resp_start < resp_end:
            response_mask[i, resp_start:resp_end] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Get per-token log-probs and optionally entropy."""
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids).logits  # (batch, seq_len, vocab)

    # Log probabilities for each token
    log_probs_all = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab)

    # Gather log probs for the label tokens
    # labels shape: (batch, seq_len)
    log_probs = log_probs_all.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result = {"log_probs": log_probs}

    if return_token_entropy:
        # Entropy: -sum(p * log_p)
        probs = torch.exp(log_probs_all)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        result["token_entropy"] = entropy

    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute rewards for rollout responses."""
    rewards = []
    format_rewards = []

    for response, gt in zip(rollout_responses, repeated_ground_truths):
        result = reward_fn(response, gt)
        rewards.append(result["reward"])
        format_rewards.append(result["format_reward"])

    raw_rewards = torch.tensor(rewards, dtype=torch.float32)
    metadata = {
        "mean_reward": float(raw_rewards.mean()),
        "mean_format_reward": float(torch.tensor(format_rewards).mean()),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute group-normalized advantages."""
    n = raw_rewards.shape[0]
    n_groups = n // group_size

    # Reshape into groups
    grouped = raw_rewards.view(n_groups, group_size)

    # Apply baseline
    if baseline == "mean":
        group_means = grouped.mean(dim=1, keepdim=True)
        advantages = grouped - group_means
    elif baseline == "none":
        advantages = grouped.clone()
    else:
        raise NotImplementedError(f"Unsupported baseline: {baseline}")

    # Apply normalizer
    if advantage_normalizer == "std":
        group_stds = grouped.std(dim=1, keepdim=True)
        advantages = advantages / (group_stds + advantage_eps)
    elif advantage_normalizer == "mean":
        group_means_abs = grouped.mean(dim=1, keepdim=True)
        advantages = advantages / (group_means_abs + advantage_eps)
    elif advantage_normalizer == "none":
        pass
    else:
        raise NotImplementedError(f"Unsupported normalizer: {advantage_normalizer}")

    advantages = advantages.view(n)

    metadata = {
        "mean_advantage": float(advantages.mean()),
        "std_advantage": float(advantages.std()),
    }
    return advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute per-token policy gradient loss."""
    batch_size, seq_len = policy_log_probs.shape

    # Ensure advantages is (batch_size, 1) for broadcasting
    adv = raw_rewards_or_advantages.view(batch_size, -1)
    if adv.shape[1] == 1:
        pass  # already (batch_size, 1)
    else:
        adv = adv.unsqueeze(-1)

    metadata: dict[str, torch.Tensor] = {}

    if importance_reweighting_method == "none":
        # On-policy: loss = -advantage * log_prob (per token)
        per_token_loss = -(adv * policy_log_probs)

    elif importance_reweighting_method == "noclip":
        # Off-policy without clipping: token-level importance weights
        # Surrogate objective: A * w_t where w_t = pi_theta/pi_0
        # per_token_loss = -A * w_t (gradient through w_t gives the PG)
        assert old_log_probs is not None
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        per_token_loss = -(adv * ratio)

    elif importance_reweighting_method == "grpo":
        # PPO/GRPO-style clipped token-level importance reweighting
        # J = min(A*w_t, A*clip(w_t, [1-eps, 1+eps]))
        assert old_log_probs is not None
        assert cliprange is not None
        log_ratio = policy_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
        obj_unclipped = adv * ratio
        obj_clipped = adv * clipped_ratio
        per_token_obj = torch.min(obj_unclipped, obj_clipped)
        per_token_loss = -per_token_obj

        clip_frac = ((ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)).float().mean()
        metadata["clip_fraction"] = clip_frac

    elif importance_reweighting_method == "gspo":
        # GSPO: sequence-level geometric mean importance weight
        # s = exp(mean(log_ratio over response tokens))
        # J = min(A*s, A*clip(s, [1-eps, 1+eps]))
        assert old_log_probs is not None
        assert cliprange is not None
        assert response_mask is not None

        log_ratio = policy_log_probs - old_log_probs
        mask_sum = response_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_log_ratio = (log_ratio * response_mask).sum(dim=1, keepdim=True) / mask_sum
        s = torch.exp(mean_log_ratio)  # (batch, 1)

        clipped_s = torch.clamp(s, 1.0 - cliprange, 1.0 + cliprange)
        obj_unclipped = adv * s
        obj_clipped = adv * clipped_s
        per_token_obj = torch.min(obj_unclipped, obj_clipped)
        per_token_loss = -per_token_obj.expand_as(policy_log_probs)

        clip_frac = ((s < 1.0 - cliprange) | (s > 1.0 + cliprange)).float().mean()
        metadata["clip_fraction"] = clip_frac

    else:
        raise NotImplementedError(f"Unsupported method: {importance_reweighting_method}")

    return per_token_loss, metadata


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: torch.Tensor,
    mask: torch.Tensor,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> torch.Tensor:
    """Aggregate per-token loss across tokens and sequences."""
    if loss_normalization == "sequence":
        # Average over tokens in each sequence (using mask), then average over sequences
        mask_sum = mask.sum(dim=1).clamp(min=1)  # (batch,)
        per_seq_loss = (per_token_policy_gradient_loss * mask).sum(dim=1) / mask_sum
        return per_seq_loss.mean()

    elif loss_normalization == "constant":
        assert normalization_constant is not None
        total_loss = (per_token_policy_gradient_loss * mask).sum()
        return total_loss / normalization_constant

    else:
        raise NotImplementedError(f"Unsupported normalization: {loss_normalization}")


def grpo_train_step(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    optimizer: torch.optim.Optimizer,
    gradient_accumulation_steps: int,
    max_grad_norm: float | None,
    reward_fn: Callable[[str, str], dict[str, float]],
    repeated_prompts: list[str],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "none", "mean"] = "std",
    importance_reweighting_method: Literal["none", "noclip", "grpo", "gspo"] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    loss_normalization: Literal["sequence", "constant"] = "sequence",
    normalization_constant: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
    """Full GRPO train step with gradient accumulation."""
    device = next(model.parameters()).device

    # 1. Compute rewards
    raw_rewards, reward_metadata = compute_rollout_rewards(
        reward_fn, rollout_responses, repeated_ground_truths
    )

    # 2. Compute advantages
    advantages, adv_metadata = compute_group_normalized_rewards(
        raw_rewards, group_size, baseline, advantage_eps, advantage_normalizer
    )

    # 3. Find non-zero advantage indices (optimization: skip zero-advantage sequences)
    nonzero_mask = advantages != 0.0
    nonzero_indices = torch.where(nonzero_mask)[0]

    if len(nonzero_indices) == 0:
        # No useful gradients; return zero loss
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        return loss, {"mean_reward": reward_metadata["mean_reward"]}

    # Filter to non-zero advantage sequences
    filtered_prompts = [repeated_prompts[i] for i in nonzero_indices]
    filtered_responses = [rollout_responses[i] for i in nonzero_indices]
    filtered_advantages = advantages[nonzero_indices]
    filtered_old_log_probs = None
    if old_log_probs is not None:
        filtered_old_log_probs = old_log_probs[nonzero_indices]

    # 4. Tokenize
    tokenized = tokenize_prompt_and_output(filtered_prompts, filtered_responses, tokenizer)
    input_ids = tokenized["input_ids"].to(device)
    labels = tokenized["labels"].to(device)
    response_mask = tokenized["response_mask"].to(device)

    # 5. Adjust gradient accumulation steps proportionally
    total_sequences = len(filtered_prompts)
    actual_grad_accum = min(gradient_accumulation_steps, total_sequences)
    microbatch_size = max(1, total_sequences // actual_grad_accum)

    # Determine scaling for gradient accumulation
    # Total batch for normalization purposes
    full_batch_size = len(repeated_prompts)

    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)
    all_entropy = []

    for i in range(0, total_sequences, microbatch_size):
        end = min(i + microbatch_size, total_sequences)
        mb_input_ids = input_ids[i:end]
        mb_labels = labels[i:end]
        mb_response_mask = response_mask[i:end]
        mb_advantages = filtered_advantages[i:end]
        mb_old_log_probs = None
        if filtered_old_log_probs is not None:
            mb_old_log_probs = filtered_old_log_probs[i:end].to(device)

        # Forward pass
        logits = model(mb_input_ids).logits
        log_probs_all = F.log_softmax(logits, dim=-1)
        mb_log_probs = log_probs_all.gather(
            dim=-1, index=mb_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Entropy
        probs = torch.exp(log_probs_all)
        entropy = -(probs * log_probs_all).sum(dim=-1)
        masked_entropy = (entropy * mb_response_mask).sum() / mb_response_mask.sum().clamp(min=1)
        all_entropy.append(masked_entropy.item())

        # Compute policy gradient loss
        per_token_loss, _ = compute_policy_gradient_loss(
            raw_rewards_or_advantages=mb_advantages,
            policy_log_probs=mb_log_probs,
            importance_reweighting_method=importance_reweighting_method,
            old_log_probs=mb_old_log_probs,
            cliprange=cliprange,
            response_mask=mb_response_mask,
        )

        # Aggregate
        mb_loss = aggregate_loss_across_microbatch(
            per_token_loss, mb_response_mask, loss_normalization, normalization_constant
        )

        # Scale for gradient accumulation
        if loss_normalization == "sequence":
            # Scale by proportion of sequences in this microbatch relative to full filtered batch
            scale = (end - i) / total_sequences
        else:
            # For constant normalization, the constant already handles it
            scale = 1.0

        scaled_loss = mb_loss * scale
        scaled_loss.backward()
        total_loss = total_loss + scaled_loss.detach()

    # Gradient clipping
    grad_norm = torch.tensor(0.0)
    if max_grad_norm is not None:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        # Compute grad norm for logging
        params_with_grad = [p for p in model.parameters() if p.grad is not None]
        if params_with_grad:
            grad_norm = torch.sqrt(
                sum(p.grad.data.norm() ** 2 for p in params_with_grad)
            )

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    metadata: dict[str, torch.Tensor | float] = {
        "grad_norm": float(grad_norm),
        "mean_entropy": float(sum(all_entropy) / max(len(all_entropy), 1)),
        **reward_metadata,
        **adv_metadata,
    }

    return total_loss, metadata



