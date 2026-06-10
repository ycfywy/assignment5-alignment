# CS336 Assignment 5: Reasoning RL - 学习文档

## 目录
- [1. 作业总览](#1-作业总览)
- [2. 背景知识](#2-背景知识)
- [3. Prompting（提示工程）](#3-prompting提示工程)
- [4. GRPO 算法实现](#4-grpo-算法实现)
- [5. RL 算法变体](#5-rl-算法变体)
- [6. Off-Policy RL](#6-off-policy-rl)
- [7. 自己设计策略梯度估计器](#7-自己设计策略梯度估计器)

---

## 1. 作业总览

### 核心目标
通过 **强化学习（RL）** 训练语言模型（OLMo-2-0425-1B）来提升其在 GSM8K 数学推理任务上的表现。

### 需要实现的内容
1. Zero-shot、few-shot 和 chain-of-thought prompting
2. **Group Relative Policy Optimization (GRPO)** 算法
3. 策略梯度估计变体（Dr. GRPO、RFT、MaxRL）
4. Off-policy RL（重要性重加权和裁剪策略）

### 需要运行的实验
1. 测量 OLMo-2-0425-1B 在 GSM8K 上的 prompting 表现
2. 运行 on-policy GRPO 提升 GSM8K 表现
3. 运行各种 RL 变体（RFT、Dr. GRPO、MaxRL）
4. 运行 off-policy GRPO 探索裁剪策略

### 使用的模型和数据
- **模型**: `allenai/OLMo-2-0425-1B`（4T tokens 预训练）
- **数据集**: GSM8K（小学数学推理题）
- **推理引擎**: vLLM
- **框架**: Hugging Face Transformers

---

## 2. 背景知识

### 核心概念

| 概念 | 语言模型术语 | RL 术语 | 含义 |
|------|-------------|---------|------|
| 𝑥 | prompt/question | initial state | 从数据集采样的问题 |
| 𝑦 | response/completion | trajectory | 模型生成的答案 |
| 𝜋θ | model | policy | 带参数θ的模型 |
| 𝑟(𝑦\|𝑥) | - | reward | 回答是否正确（0或1） |
| 𝐵 | prompts per batch | - | 每批的问题数 |
| 𝐺 | generations per prompt | group size | 每个问题的生成数 |

### 优化目标
```
J_θ = E_{x~ρ} E_{y~π_θ(y|x)} [r(y|x)]
```
即最大化模型在数学题上的**期望正确率**。

---

## 3. Prompting（提示工程）

### 讲了什么
- 基础 prompting 策略：question_only、r1_zero、r1_zero_three_shot
- vLLM 推理引擎的使用
- 答案评分函数（grader）的设计

### 需要实现什么

#### Problem: prompting_baselines（5分）
- 写脚本评估 3 种 prompt 策略在 GSM8K 上的表现
- 观察模型生成的 format reward 和 correctness reward 分布
- 分析模型行为差异

### 实现步骤
1. 加载 vLLM 服务器（使用 `cs336_alignment/vllm_utils.py`）
2. 读取 `data/gsm8k/test.jsonl`
3. 用 3 种 prompt 模板格式化问题：
   - `prompts/question_only.prompt`: `{question} Please put your final answer within \boxed{}.`
   - `prompts/r1_zero.prompt`: 带 `<think>` / `<answer>` 标签的推理格式
   - `prompts/r1_zero_three_shot_gsm8k.prompt`: 三样本版本
4. 使用 `drgrpo_grader.py` 中的 `r1_zero_reward_fn` / `question_only_reward_fn` 评分
5. 生成参数：temperature=1.0, top_p=1.0, max_tokens=512

---

## 4. GRPO 算法实现

### 讲了什么
- REINFORCE 策略梯度推导
- Baseline 减方差技术（group mean baseline）
- Advantage 标准化（除以 group std）
- 序列长度归一化
- 完整 on-policy GRPO 算法（Algorithm 1）

### 需要实现什么（按顺序）

#### 4.1 Problem: tokenize_prompt_and_output（1分）
```python
def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    # 返回 {input_ids, labels, response_mask}
```
- 分别 tokenize prompt 和 output（不加特殊 token）
- 直接拼接 prompt_ids + response_ids
- 构建 response_mask（标记哪些 label 属于 response）
- **测试**: `uv run pytest -k test_tokenize_prompt_and_output`

#### 4.2 Problem: get_response_log_probs（1分）
```python
def get_response_log_probs(model, input_ids, labels, return_token_entropy=False):
    # 返回 {"log_probs": ..., "token_entropy": ...}
```
- 计算 per-token 条件 log 概率
- 可选计算 per-token 熵
- **测试**: `uv run pytest -k test_get_response_log_probs`

#### 4.3 Problem: compute_rollout_rewards（1分）
```python
def compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths):
    # 返回 (raw_rewards, metadata)
```
- 调用 reward_fn 计算每个 rollout 的奖励
- **测试**: `uv run pytest -k compute_rollout_rewards`

#### 4.4 Problem: compute_group_normalized_rewards（1分）
```python
def compute_group_normalized_rewards(raw_rewards, group_size, baseline="mean",
                                      advantage_eps=1e-6, advantage_normalizer="std"):
    # 返回 (advantages, metadata)
```
- 减去 group mean（baseline）
- 除以 group std（normalizer）+ eps
- **测试**: `uv run pytest -k compute_group_normalized_rewards_grpo`

#### 4.5 Problem: compute_policy_gradient_loss（1分）
```python
def compute_policy_gradient_loss(raw_rewards_or_advantages, policy_log_probs,
                                  importance_reweighting_method="none", ...):
    # 返回 (per_token_loss, metadata)
```
- 计算 per-token 策略梯度损失
- 返回**负目标**（因为 PyTorch 做梯度下降）
- **测试**: `uv run pytest -k test_compute_policy_gradient_loss_on_policy`

#### 4.6 Problem: aggregate_loss_across_microbatch（0.5分）
```python
def aggregate_loss_across_microbatch(per_token_loss, mask,
                                      loss_normalization="sequence", ...):
    # 返回 scalar loss
```
- "sequence": 先对每个序列求平均，再对序列间求平均
- **测试**: `uv run pytest -k test_aggregate_loss_across_microbatch_sequence`

#### 4.7 Problem: grpo_train_step（5分）
```python
def grpo_train_step(model, tokenizer, optimizer, gradient_accumulation_steps,
                    max_grad_norm, reward_fn, repeated_prompts, rollout_responses,
                    repeated_ground_truths, group_size, ...):
    # 返回 (loss, metadata)
```
- 串联以上所有组件
- 实现梯度累积
- 梯度裁剪
- 记录 metrics（loss, grad_norm, entropy, rewards）
- **测试**: `uv run pytest -k test_grpo_train_step_standard_on_policy`

#### 4.8 Problem: baseline_calcs（5分，数学证明）
- 计算策略梯度估计器的方差
- 比较有无 baseline 的方差

#### 4.9 Problem: grpo_experiments（10分，实验）
- 建议超参数：
  - n_train=6400, n_val=1024
  - rollout_batch=train_batch=256, group_size=8
  - lr=1e-5, grad_accum=32, max_grad_norm=1.0
  - 4 个随机种子
- 目标：validation accuracy ≥ 25%

#### 4.10 Problem: grpo_learning_rate（3分）
- Learning rate sweep

#### 4.11 Problem: grpo_prompt_ablation（3分）
- 比较 3 种 prompt 的 RL 训练效果

---

## 5. RL 算法变体

### 讲了什么
- **Dr. GRPO**: 去掉 std 归一化 + 常数归一化（而非序列长度归一化）
- **RFT (Rejection Fine-Tuning)**: 只在正确回答上做 SFT
- **MaxRL**: 用 group mean 代替 group std 做归一化

### 需要实现什么

#### Problem: compute_group_normalized_rewards_drgrpo（0.5分）
- 支持 `advantage_normalizer="none"` 和 `baseline="none"`
- **测试**: `uv run pytest -k compute_group_normalized_rewards_drgrpo`

#### Problem: aggregate_loss_across_microbatch_constant（0.5分）
- 支持 `loss_normalization="constant"`（除以 B*G*L）
- **测试**: `uv run pytest -k test_aggregate_loss_across_microbatch_constant`

#### Problem: compute_group_normalized_rewards_maxrl（0.5分）
- 支持 `advantage_normalizer="mean"`
- **测试**: `uv run pytest -k compute_group_normalized_rewards_maxrl`

#### Problem: grpo_train_step_variants_on_policy（2.5分）
- 支持全部 on-policy 变体组合
- 跳过 zero-advantage 序列（加速训练）
- **测试**: `uv run pytest -k test_grpo_train_step_variants_on_policy`

#### Problem: grpo_experiments_variants_on_policy（10分，实验）
- 对比 GRPO_constant, Dr_GRPO, RFT, MaxRL
- 4 个随机种子

#### 数学证明题
- `think_about_length_normalization`（1分）
- `think_about_rft`（2分）
- `derive_difficulty_reweightings`（6分）
- `think_about_advantage_normalization`（2分）

---

## 6. Off-Policy RL

### 讲了什么
- 重要性重加权（Importance Reweighting）的原理
- Token-level vs Sequence-level 重加权的权衡
- PPO/GRPO 风格裁剪
- GSPO 风格序列级裁剪（几何平均）

### 需要实现什么

#### Problem: compute_policy_gradient_loss_off_policy（1分）
- 支持 `importance_reweighting_method="noclip"` 和 `"grpo"`
- **测试**: `uv run pytest -k test_compute_policy_gradient_loss_off_policy`

#### Problem: compute_policy_gradient_loss_off_policy_gspo（1分）
- 支持 `importance_reweighting_method="gspo"`（几何平均 + 裁剪）
- **测试**: `uv run pytest -k test_compute_policy_gradient_loss_off_policy_gspo`

#### Problem: grpo_train_step_off_policy（2.5分）
- 训练步支持 off-policy 参数
- **测试**: `uv run pytest -k test_grpo_train_step_off_policy`

#### Problem: grpo_experiments_off_policy（10分，实验）
- 对比：offpolicy_naive, offpolicy_noclip, offpolicy_clip, offpolicy_gspo
- rollout_batch=256, train_batch=8, grad_accum=1（32x off-policy）
- cliprange: GRPO=0.2, GSPO=3e-4

#### 数学证明题
- `derive_surrogate_objectives`（2分）
- `think_about_importance_reweighting`（2分）

---

## 7. 自己设计策略梯度估计器

### Problem: try_your_own（10分）
- 基于已有方法修改一个组件
- 与 baseline 对比
- 提供理论依据
- 可选方向：不同 advantage 估计器、不同重要性重加权策略、不同 reward baseline

---

## 实现路线图

### Phase 1: 基础组件（测试驱动）
1. `tokenize_prompt_and_output` → 过 test
2. `get_response_log_probs` → 过 test
3. `compute_rollout_rewards` → 过 test
4. `compute_group_normalized_rewards`（支持 mean/std）→ 过 test
5. `compute_policy_gradient_loss`（on-policy, method="none"）→ 过 test
6. `aggregate_loss_across_microbatch`（sequence）→ 过 test
7. `grpo_train_step`（standard on-policy）→ 过 test

### Phase 2: 变体扩展
8. `compute_group_normalized_rewards` 扩展（none, mean normalizer + none baseline）
9. `aggregate_loss_across_microbatch` 扩展（constant）
10. `grpo_train_step` 支持全部 on-policy 变体

### Phase 3: Off-policy 扩展
11. `compute_policy_gradient_loss` 扩展（noclip, grpo, gspo）
12. `grpo_train_step` 支持 off-policy

### Phase 4: 实验和写作
13. Prompting baseline 实验
14. On-policy GRPO 实验（4 seeds × 200 steps）
15. Learning rate sweep
16. Prompt ablation
17. 变体对比实验
18. Off-policy 实验
19. 自定义策略梯度
20. 完成数学证明和 writeup

---

## 关键公式速查

### REINFORCE 策略梯度
```
∇_θ J_θ = E_{x~ρ} E_{y~π_θ} [r(y|x) ∇_θ log π_θ(y|x)]
```

### On-Policy GRPO 损失
```
J = (1/BG) Σ_i Σ_j (1/len(y)) Σ_t A(i,j) · log π_θ(y_t | x, y<t)
```
其中 `A(i,j) = (r(y(i,j)|x(i)) - μ_i) / std_i`

### Dr. GRPO
```
g = (1/Z) Σ Σ Σ_t (r - μ_i) ∇_θ log π_θ(y_t | ...)
```
Z = B × G × L (常数归一化)

### Off-Policy GRPO (with clipping)
```
J = (1/BG) Σ_i Σ_j (1/len(y)) Σ_t min(A·w_t, A·clip(w_t, [1-ε,1+ε]))
```
其中 `w_t = π_θ(y_t|...)/π_0(y_t|...)`

### GSPO (序列级)
```
s = (Π_t w_t)^(1/len(y))  # 几何平均
J = (1/BG) Σ_i Σ_j min(A·s, A·clip(s, [1-ε,1+ε]))
```
