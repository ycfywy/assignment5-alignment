# Assignment 5 实现总结

## 测试结果
**26/26 全部通过** ✅

```
tests/test_grpo.py       19 passed ✅  (GRPO全部组件 + 训练步)
tests/test_data.py        2 passed ✅  (SFT数据集 + 批次迭代器)
tests/test_metrics.py     4 passed ✅  (MMLU/GSM8K解析)
tests/test_dpo.py         1 passed ✅  (DPO损失)
```

## 新建文件

### 1. `cs336_alignment/grpo.py` — GRPO 核心实现
包含以下函数（全部通过测试）：

| 函数 | 说明 |
|------|------|
| `tokenize_prompt_and_output` | 分别 tokenize prompt/output，拼接，构建 response_mask |
| `get_response_log_probs` | 从 causal LM 获取 per-token 条件 log-prob 和 entropy |
| `compute_rollout_rewards` | 调用 reward_fn 计算每个 rollout 的奖励 |
| `compute_group_normalized_rewards` | 组内归一化（支持 mean/none baseline + std/none/mean normalizer）|
| `compute_policy_gradient_loss` | 策略梯度损失（支持 none/noclip/grpo/gspo 四种重加权）|
| `aggregate_loss_across_microbatch` | 聚合损失（支持 sequence/constant 归一化）|
| `grpo_train_step` | 完整训练步（含梯度累积、裁剪、zero-advantage 跳过）|

**支持的算法变体：**
- Standard GRPO (on-policy, sequence normalization)
- GRPO with constant normalization
- Dr. GRPO (no std normalization + constant)
- RFT (no baseline, no normalization + constant)
- MaxRL (mean normalization + constant)
- Off-policy noclip (token-level importance reweighting)
- Off-policy GRPO (PPO-style clipped token-level reweighting)
- Off-policy GSPO (sequence-level geometric mean reweighting with clipping)

### 2. `cs336_alignment/metrics.py` — 评估指标解析
| 函数 | 说明 |
|------|------|
| `parse_mmlu_response` | 从模型输出提取 A/B/C/D 答案字母 |
| `parse_gsm8k_response` | 从模型输出提取最后一个数字作为答案 |

### 3. `cs336_alignment/sft_data.py` — SFT 数据加载
| 类/函数 | 说明 |
|---------|------|
| `PackedSFTDataset` | PyTorch Dataset，将 JSONL 数据 tokenize、拼接并切成定长块 |
| `iterate_batches` | 封装 DataLoader 返回批次迭代器 |

### 4. `cs336_alignment/dpo.py` — DPO 损失
| 函数 | 说明 |
|------|------|
| `compute_per_instance_dpo_loss` | 计算单个 (prompt, chosen, rejected) 的 DPO 损失 |

### 5. `tests/adapters.py` — 测试适配器
将所有实现连接到测试框架。

---

## 需要训练/GPU 才能完成的部分（未实现）

以下部分需要 GPU 和实际模型训练，未在本次实现中完成：

### 主作业 (Reasoning RL)
1. **prompting_baselines** — 需要加载 OLMo-2-0425-1B + vLLM 推理
2. **grpo_experiments_standard_on_policy** — GRPO 训练循环（4 seeds × 200 steps）
3. **grpo_learning_rate** — LR sweep
4. **grpo_prompt_ablation** — 3 种 prompt 对比
5. **grpo_experiments_variants_on_policy** — 变体对比（GRPO/Dr.GRPO/RFT/MaxRL）
6. **grpo_experiments_off_policy** — Off-policy 实验
7. **try_your_own** — 自定义策略梯度估计器

### 补充作业 (SFT + DPO)
1. **Zero-shot 评估** — 需要 Llama 3.1 8B + vLLM（MMLU/GSM8K/AlpacaEval/SST）
2. **SFT 训练** — fine-tune Llama 3.1 8B（~3 GPU hrs）
3. **SFT 评估** — 同上评估
4. **DPO 训练** — 在 HH 数据上训练（~1 GPU hr）
5. **DPO 评估** — 评估 DPO 模型
6. **Red-teaming** — 交互式测试

### 数学证明题
- baseline_calcs（方差计算）
- think_about_length_normalization
- think_about_rft
- derive_difficulty_reweightings
- think_about_advantage_normalization
- derive_surrogate_objectives
- think_about_importance_reweighting

---

## 如何运行测试
```bash
cd /root/aigame/dannyyan/cs336/assignment5-alignment
/root/aigame/.tools/uv run pytest tests/ -v
```
