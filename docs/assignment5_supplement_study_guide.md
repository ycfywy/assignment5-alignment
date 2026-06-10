# CS336 Assignment 5 Supplement: Instruction Tuning & RLHF - 学习文档

## 目录
- [1. 总览](#1-总览)
- [2. 动机：训练通用 LLM](#2-动机训练通用-llm)
- [3. Zero-Shot 评估](#3-zero-shot-评估)
- [4. Instruction Fine-Tuning (SFT)](#4-instruction-fine-tuning-sft)
- [5. 评估 SFT 模型](#5-评估-sft-模型)
- [6. DPO 对齐](#6-dpo-对齐)
- [实现路线图](#实现路线图)

---

## 1. 总览

### 核心目标
将 **Llama 3.1 8B** 从一个基础语言模型训练成一个能安全、有帮助地遵循指令的通用对话助手。

### 需要实现的内容
1. Zero-shot prompting 基线（4 个评估任务）
2. **Supervised Fine-Tuning (SFT)**：使用 instruction-response 数据微调
3. **Direct Preference Optimization (DPO)**：从偏好数据学习对齐

### 需要运行的实验
1. 测量 Llama 3.1 8B 的 zero-shot 表现
2. Instruction fine-tune Llama 3.1 8B
3. 用偏好数据做 DPO fine-tune

### 评估基准
| 基准 | 评测维度 | 指标 |
|------|----------|------|
| MMLU | 事实知识 | 准确率（A/B/C/D选择题） |
| GSM8K | 数学推理 | 最终数字准确率 |
| AlpacaEval | 对话质量 | Win rate vs GPT-4 Turbo |
| SimpleSafetyTests | 安全性 | 安全回复比例 |

### 模型和基础设施
- **基础模型**: `meta-llama/Meta-Llama-3.1-8B`
- **评判模型**: `meta-llama/Llama-3.3-70B-Instruct`
- **平台**: Modal 云计算
- **共享卷**: `/mnt/cs336-a5-supplement`（模型权重 + 数据）

---

## 2. 动机：训练通用 LLM

### 讲了什么
- 目标是构建**通用对话系统**（而非特定推理任务）
- 训练流程：Base Model → SFT → DPO
- 用 4 个代表性下游任务评估模型能力变化

---

## 3. Zero-Shot 评估

### 讲了什么
- 使用统一系统 prompt（`prompts_safety/zero_shot_system_prompt.prompt`）
- 为每个 benchmark 设计特定任务 prompt
- 生成参数：贪心解码（temperature=0.0, top_p=1.0）

### 系统 Prompt 结构
```
# Instruction
Below is a list of conversations between a human and an AI assistant (you).
Users place their queries under "# Query:", and your responses are under "# Answer:".
You are a helpful, respectful, and honest assistant...

# Query:
```{instruction}```

# Answer:
```

### 需要实现什么

#### 3.1 Problem: mmlu_baseline（4分）
**需要实现：**
- (a) 解析 MMLU 模型输出的函数（提取 A/B/C/D）
  - 实现 `run_parse_mmlu_response` in `tests/adapters.py`
  - **测试**: `uv run pytest -k test_parse_mmlu_response`
- (b) 评估脚本（加载数据→格式化prompt→生成→计算指标→序列化）
- (c-f) 运行评估 + 分析

**MMLU 任务 Prompt** (`prompts_safety/mmlu_zero_shot.prompt`):
```
Answer the following multiple choice question about {subject}. Respond with a single
sentence of the form "The correct answer is _", filling the blank with the letter
corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer:
```

#### 3.2 Problem: gsm8k_baseline（4分）
**需要实现：**
- (a) 解析 GSM8K 输出的函数（提取最后一个数字）
  - 实现 `run_parse_gsm8k_response` in `tests/adapters.py`
  - **测试**: `uv run pytest -k test_parse_gsm8k_response`
- (b) 评估脚本
- (c-f) 运行评估 + 分析

**GSM8K 任务 Prompt** (`prompts_safety/gsm8k_zero_shot.prompt`):
```
{question}
Answer:
```

#### 3.3 Problem: alpaca_eval_baseline（4分）
**需要实现：**
- (a) 生成脚本（输出格式：JSON array with instruction/output/generator/dataset）
- (b-d) 运行 + AlpacaEval 评估

**AlpacaEval 任务 Prompt** (`prompts_safety/alpaca_eval_zero_shot.prompt`):
```
{instruction}
```

#### 3.4 Problem: sst_baseline（4分）
**需要实现：**
- (a) 生成脚本（输出格式：JSONL with prompts_final/output）
- (b-d) 运行 + 安全评估

---

## 4. Instruction Fine-Tuning (SFT)

### 讲了什么
- SFT 原理：在 (prompt, response) 对上做语言建模
- 数据来源：UltraChat-200K + SafetyTunedLlamas
- SFT Prompt 模板（Alpaca format）
- 数据打包（packing into constant-length sequences）
- 梯度累积技术
- 模型加载（bfloat16 + FlashAttention-2）

### SFT 训练格式（`prompts_safety/alpaca_sft.prompt`）
```
Below is an instruction that describes a task. Write a response that appropriately
completes the request.

### Instruction:
{instruction}

### Response:
{response}
```

### 需要实现什么

#### 4.1 Problem: look_at_sft（4分）
- 查看 10 个随机训练样本，分析数据质量

#### 4.2 Problem: data_loading（3分）
**(a) PyTorch Dataset 子类：**
```python
class PackedSFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, seq_length, shuffle): ...
    def __len__(self): ...  # 返回序列数
    def __getitem__(self, i): ...  # 返回 {input_ids, labels}
```
- 数据格式：gzip 压缩的 JSONL（每行有 prompt/response 字段）
- 将所有文档 tokenize 后拼接（EOS 分隔）
- 切成 seq_length 大小的非重叠块
- 实现 `get_packed_sft_dataset` in adapters
- **测试**: `uv run pytest -k test_packed_sft_dataset`

**(b) 批次迭代器：**
- 实现 `run_iterate_batches` in adapters
- **测试**: `uv run pytest -k test_iterate_batches`

#### 4.3 Problem: sft_script（4分）
**训练脚本需要：**
- 加载 Llama 3.1 8B（bfloat16 + FlashAttention-2）
- 前向传播 + 交叉熵损失
- 梯度累积
- 周期性验证
- `model.save_pretrained()` / `tokenizer.save_pretrained()`

**不能使用** Hugging Face `Trainer` 类。

#### 4.4 Problem: sft（6分，需 3 B200 hrs）
**推荐超参数：**
- 1 epoch, context_length=512, batch_size=32
- lr=2e-5, cosine decay, warmup=3%
- weight_decay=0.1, grad_clip=1.0

---

## 5. 评估 SFT 模型

### 讲了什么
- 评估时使用 Alpaca SFT prompt（而非 zero-shot system prompt）
- 对比 SFT vs Zero-shot baseline 的表现变化

### 需要实现什么

| Problem | 分值 | 内容 |
|---------|------|------|
| mmlu_sft | 4分 | MMLU 评估 + 对比 |
| gsm8k_sft | 4分 | GSM8K 评估 + 对比 |
| alpaca_eval_sft | 4分 | AlpacaEval 评估 + 对比 |
| sst_sft | 4分 | 安全测试评估 + 对比 |
| red_teaming | 4分 | Red-team 你的 SFT 模型 |

---

## 6. DPO 对齐

### 讲了什么
- RLHF 的复杂性：奖励模型 → PPO → 高度不稳定
- DPO 的简化：直接从偏好数据优化，无需显式奖励模型
- DPO 损失公式推导

### DPO 核心公式
```
ℓ_DPO(π_θ, π_ref, x, y_w, y_l) = -log σ(β·(log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))))
```
- `y_w`: 偏好的（chosen）回复
- `y_l`: 被拒绝的（rejected）回复
- `π_ref`: 参考模型（SFT 后的模型，冻结不动）
- `β`: 控制偏离参考模型的惩罚力度

### 需要实现什么

#### 6.1 Problem: look_at_hh（2分）
- (a) 加载 Anthropic HH 数据集（4个文件合并）
  - 忽略多轮对话
  - 分离 instruction / chosen / rejected
- (b) 分析 3 个 helpful + 3 个 harmless 样本

#### 6.2 Problem: dpo_loss（2分）
```python
def compute_per_instance_dpo_loss(model, ref_model, tokenizer, 
                                   prompt, chosen, rejected, beta):
    # 返回 per-instance DPO loss
```
- 用 Alpaca 模板格式化
- EOS 追加在每个 response 后
- 实现 `run_compute_per_instance_dpo_loss` in adapters
- **测试**: `uv run pytest -k test_per_instance_dpo_loss`

#### 6.3 Problem: dpo_training（4分，需 1 B200 hr）
**(a) DPO 训练循环：**
- 2 GPU：一个放参考模型，一个放训练模型
- 有效 batch size = 64
- β = 0.1, lr = 1e-6
- 优化器：RMSprop（不用 AdamW）
- 验证指标：分类准确率（chosen 的 log-prob > rejected 的 log-prob）
- 保存最高验证准确率的 checkpoint

**(b-d) 评估：**
- AlpacaEval win rate
- SimpleSafetyTests 安全率
- GSM8K 和 MMLU（观察 alignment tax）

---

## 实现路线图

### Phase 1: 评估基础设施（≈2天）
1. ✅ 实现 `parse_mmlu_response`（提取 A/B/C/D）
2. ✅ 实现 `parse_gsm8k_response`（提取最后数字）
3. 编写 zero-shot 评估脚本（4 个 benchmarks）
4. 运行 baseline + 收集结果

### Phase 2: SFT 训练（≈2天）
5. 实现 `PackedSFTDataset`
6. 实现 batch 迭代器
7. 编写 SFT 训练脚本
8. 训练 1 epoch（≈3 GPU hrs）
9. 保存模型

### Phase 3: SFT 评估（≈1天）
10. 用 SFT 模型跑 4 个 benchmarks
11. 对比分析
12. Red-teaming

### Phase 4: DPO（≈2天）
13. 加载 HH 数据 + 预处理
14. 实现 DPO loss → 过 test
15. 编写 DPO 训练循环
16. 训练 1 epoch（≈1 GPU hr）
17. 评估 DPO 模型

---

## 关键技术要点

### 模型加载代码
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

### 梯度累积模板
```python
gradient_accumulation_steps = 4
for idx, (inputs, labels) in enumerate(data_loader):
    logits = model(inputs)
    loss = loss_fn(logits, labels) / gradient_accumulation_steps
    loss.backward()
    
    if (idx + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### DPO Loss 计算关键点
1. 分别计算 `log π_θ(y_w|x)` 和 `log π_θ(y_l|x)`
2. 分别计算 `log π_ref(y_w|x)` 和 `log π_ref(y_l|x)`
3. `log_ratio_w = log_π_θ(y_w) - log_π_ref(y_w)`
4. `log_ratio_l = log_π_θ(y_l) - log_π_ref(y_l)`
5. `loss = -log(sigmoid(β * (log_ratio_w - log_ratio_l)))`

### 生成参数
| 场景 | Temperature | Top-p | Max tokens |
|------|-------------|-------|------------|
| Zero-shot 评估 | 0.0 | 1.0 | - |
| RL 训练 (main) | 1.0 | 1.0 | 512 |
