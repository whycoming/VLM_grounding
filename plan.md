# 从第一性原理优化 VLM 视觉 Grounding：诊断驱动的 Post-Training 研究

> **核心命题**：VLM 视觉 Grounding 的 post-training 不应该默认选 GRPO——而应该先诊断失败模式，再选择匹配的训练方法。本项目系统对比 SFT、RFT、Online DPO、GRPO、DAPO 在 visual grounding 上的效果与效率，基于诊断结果提出最优组合 pipeline。
>
> **硬件约束**：单张 RTX 4090（24GB VRAM）
>
> **预计周期**：8–10 周
>
> **最终产出**：开源代码 + 模型权重 + 系统对比论文/技术博客

---

## 第一性原理分析

### Visual Grounding 的本质分解

模型需要完成三个子任务，每个子任务的失败模式和最优训练信号完全不同：

```
输入：图像 I + 文本描述 T（如 "the red cup to the left of the tall bottle"）
输出：边界框 [x1, y1, x2, y2]

子任务 1：语义理解 — 解析 T 中的属性（red）、类别（cup）、空间关系（left of）
子任务 2：视觉定位 — 在 I 中找到候选区域并消歧
子任务 3：坐标输出 — 将视觉定位结果精确映射为数值坐标
```

### 为什么不应该无脑选 GRPO

| 因素 | 数学推理（GRPO 的主场） | Visual Grounding |
|------|------------------------|------------------|
| 输出长度 | 数百 token 推理链 | 极短（4 个数字） |
| Reward 性质 | 二值（对/错） | 连续（IoU ∈ [0,1]） |
| 探索空间 | 巨大（多种推理路径） | 有限（坐标微调） |
| GRPO 优势来源 | 发现新推理路径 | 被削弱 |
| 单卡瓶颈 | 可接受（文本采样快） | 严重（视觉编码重复计算） |

GRPO 的核心价值是通过 group 内多样性的 exploration 发现更好的策略。但 grounding 的输出空间太小，8 个 sample 可能只是 8 组略有不同的坐标——缺乏有意义的多样性。

### 研究假设

**H1**：对于 visual grounding，诊断失败模式后选择针对性方法，优于盲目使用 GRPO。

**H2**：简单方法（RFT、Online DPO）在 grounding 上的 性能/计算成本 比 高于 GRPO/DAPO。

**H3**：分阶段 pipeline（SFT → 针对性强化 → RL 精调）优于任何单一方法。

---

## Phase 0: 环境搭建与数据准备（Day 1–3）

### 0.1 环境

```bash
conda create -n vlm-grounding python=3.10
conda activate vlm-grounding

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/trl.git
pip install transformers accelerate peft bitsandbytes
pip install qwen-vl-utils[decord] flash-attn --no-build-isolation
pip install deepspeed wandb matplotlib seaborn

# VLM-R1 代码库（用于 GRPO/DAPO 训练）
git clone https://github.com/om-ai-lab/VLM-R1.git
```

### 0.2 模型

```bash
# 主力模型
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./models/qwen25vl-7b

# 快速验证用
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/qwen25vl-3b
```

### 0.3 数据

```bash
# COCO Train2014
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d ./data/coco/

# RefCOCO/+/g 标注（VLM-R1 格式的 JSONL）
# 从 VLM-R1 仓库获取预处理好的标注文件

# LISA-Grounding（out-of-domain 评估）
# 按照 VLM-R1 README 下载
```

### 0.4 验证检查点

- [ ] 模型可正常加载并推理
- [ ] 数据集完整，JSONL 格式正确
- [ ] wandb 登录成功

---

## Phase 1: 诊断性分析（Day 4–10）—— 本项目的基石

### 1.1 目标

**不是为了刷分，而是为了理解 VLM 在 grounding 上到底哪里失败。** 这个阶段的产出直接决定后续所有方法的选择。

### 1.2 Zero-shot 评估

在 RefCOCO/+/g 全部 split + LISA-Grounding 上评估 Qwen2.5-VL-7B 的 zero-shot 表现。

```python
# eval_baseline.py
# 指标：Acc@0.5, Acc@0.75, Mean IoU
# 在每个 split 上分别计算
```

### 1.3 失败模式分类（核心）

对 **所有预测错误的样本**（IoU < 0.5）进行系统分类：

```python
# diagnose.py

def classify_failure(pred_box, gt_box, image, description, model_output):
    """
    将每个失败案例分类到以下类别之一
    """
    iou = compute_iou(pred_box, gt_box)

    # Type A: 格式错误 — 模型没有输出合法的坐标
    if pred_box is None:
        return "FORMAT_ERROR"

    # Type B: 坐标偏移 — 找对了目标但坐标不准
    # 判据：pred 和 gt 的中心点距离 < gt 对角线的 30%，但 IoU < 0.5
    center_dist = distance(center(pred_box), center(gt_box))
    gt_diag = diagonal(gt_box)
    if center_dist < 0.3 * gt_diag and iou < 0.5:
        return "COORDINATE_IMPRECISION"

    # Type C: 找错目标 — 定位到了图中另一个同类物体
    # 判据：pred 和 gt 的 IoU < 0.1，但 pred 与图中某个其他 object 重合
    if iou < 0.1:
        return "WRONG_TARGET"

    # Type D: 空间关系推理错误 — 描述中有空间词
    spatial_words = ["left", "right", "above", "below", "near", "between", "behind", "front"]
    if any(w in description.lower() for w in spatial_words) and iou < 0.5:
        return "SPATIAL_REASONING_ERROR"

    # Type E: 属性理解错误
    return "ATTRIBUTE_ERROR"
```

### 1.4 诊断报告

输出一份诊断报告，包含：

```
1. 各失败类型的占比分布（饼图）
2. 各失败类型在不同数据集上的分布差异
3. 失败案例的 referring expression 长度分布
4. 每种失败类型的 10 个代表性可视化样本
5. 结论：主要瓶颈是什么？
```

### 1.5 诊断结果 → 方法选择的映射

| 主要失败类型 | 瓶颈分析 | 推荐方法 |
|-------------|---------|---------|
| FORMAT_ERROR 占比高 | 模型不熟悉坐标输出格式 | SFT 足矣 |
| COORDINATE_IMPRECISION 占比高 | 能找到目标但坐标不准 | SFT + L1 辅助 loss，或 RFT |
| WRONG_TARGET 占比高 | 消歧能力不足 | Hard Negative DPO |
| SPATIAL_REASONING_ERROR 占比高 | 空间推理能力弱 | Thinking + DAPO |
| ATTRIBUTE_ERROR 占比高 | 视觉-语言对齐不足 | 可能需要解冻 ViT |

### 1.6 验证检查点

- [ ] Zero-shot baseline 结果完整
- [ ] 失败模式分类完成，有饼图和统计
- [ ] 每种失败类型有 10+ 可视化样本
- [ ] 诊断报告写成 `results/diagnostic_report.md`
- [ ] 基于诊断结果，确定 Phase 3 的方法优先级

---

## Phase 2: SFT Baseline + 二次诊断（Day 11–16）

### 2.1 目标

建立 SFT baseline，同时观察 SFT 能修复哪些诊断问题、留下哪些残余瓶颈。

### 2.2 训练配置

```yaml
model: Qwen2.5-VL-7B-Instruct
method: LoRA (r=64, alpha=128)
target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]
freeze_vision_tower: true
learning_rate: 1e-5
epochs: 3
batch_size: 2
gradient_accumulation: 8
gradient_checkpointing: true
bf16: true
# 显存预算：~23 GB
```

### 2.3 SFT 后二次诊断

训练完 SFT 后，重新跑 Phase 1.3 的失败模式分类：

```
对比项：
- SFT 修复了哪些失败类型？（预期：FORMAT_ERROR 大幅下降）
- SFT 没修复什么？（预期：WRONG_TARGET, SPATIAL_REASONING 残留）
- SFT 引入了什么新问题？（预期：out-of-domain 泛化下降）
```

二次诊断的结果告诉你 **SFT 之后的残余瓶颈**，精确指导 Phase 3 的方法选择。

### 2.4 验证检查点

- [ ] SFT 训练完成，loss 收敛
- [ ] 全面评估 + 二次诊断完成
- [ ] 明确 SFT 后的残余瓶颈

---

## Phase 3: 五种 Post-Training 方法系统对比（Day 17–40）—— 核心实验

### 3.0 实验设计原则

**控制变量**：所有方法使用相同基座（Phase 2 的 SFT checkpoint）、相同数据、相同 LoRA 配置、相同训练 token 预算。唯一变量是训练算法。

**公平对比**：固定 GPU 小时数，对比相同计算预算下的性能。

---

### 3.1 方法 A：Rejection Sampling Fine-Tuning（RFT）

**原理**：采样 → 按 IoU 过滤 → 对好样本做 SFT → 迭代。最简单的"从好样本中学习"。

```python
# train_rft.py

def rft_iteration(model, dataset, threshold=0.7, num_samples=16):
    good_samples = []
    for prompt in dataset:
        # 1. 采样（离线，可用 4-bit 推理，不受显存限制）
        responses = model.generate(prompt, num_return_sequences=num_samples,
                                   temperature=0.9, do_sample=True)
        # 2. 按 IoU 过滤
        for resp in responses:
            pred_box = parse_bbox(resp)
            iou = compute_iou(pred_box, prompt["gt_box"])
            if iou >= threshold:
                good_samples.append({"prompt": prompt, "response": resp})
        # 3. 对过滤后数据做 SFT
    sft_train(model, good_samples)
    return model

# 迭代 3 轮，阈值递减
for i in range(3):
    model = rft_iteration(model, train_data, threshold=0.7 - i * 0.05)
```

**为什么在 grounding 上可能很有效**：
- IoU 连续可计算，过滤阈值精确可控
- 采样和训练分离 → 显存需求远低于 GRPO
- 本质是 GRPO 的简化版，去掉 advantage normalization
- 4090 上采样用 4-bit，训练用 bf16 LoRA

**超参**：num_samples=16, threshold=0.7→0.65→0.6, iterations=3

---

### 3.2 方法 B：Online DPO

**原理**：在线生成候选对，按 IoU 排序做 chosen/rejected pair，迭代 DPO。

```python
# train_online_dpo.py

def generate_preference_pairs(model, dataset, num_samples=8):
    pairs = []
    for prompt in dataset:
        responses = model.generate(prompt, num_return_sequences=num_samples,
                                   temperature=0.9, do_sample=True)
        scored = [(resp, compute_iou(parse_bbox(resp), prompt["gt_box"]))
                  for resp in responses]
        scored.sort(key=lambda x: x[1], reverse=True)

        if scored[0][1] > scored[-1][1] + 0.1:  # 需要有足够差距
            pairs.append({
                "prompt": prompt,
                "chosen": scored[0][0],
                "rejected": scored[-1][0],
            })
    return pairs

for epoch in range(3):
    pairs = generate_preference_pairs(model, train_data)
    dpo_train(model, pairs, beta=0.1)
```

**为什么在 grounding 上有优势**：
- 只需 pair-wise 对比，显存 << GRPO
- IoU 提供天然 preference 排序
- 消歧类错误（WRONG_TARGET）特别适合 DPO

**超参**：num_samples=8, beta=0.1, min_iou_gap=0.1

---

### 3.3 方法 C：Hard Negative DPO（针对消歧）

**原理**：不随机采样，而是用标注中同图同类的其他物体构造 hard negative pair。

```python
# train_hard_neg_dpo.py

def construct_hard_negatives(dataset, coco_annotations):
    pairs = []
    for sample in dataset:
        image_id = sample["image_id"]
        gt_box = sample["gt_box"]
        gt_category = sample["category"]

        # 找同图同类的其他物体
        others = [obj for obj in coco_annotations[image_id]
                  if obj["category"] == gt_category
                  and compute_iou(obj["bbox"], gt_box) < 0.3]

        for neg in others:
            pairs.append({
                "prompt": sample,
                "chosen": format_bbox(gt_box),
                "rejected": format_bbox(neg["bbox"]),
            })
    return pairs
```

**为什么可能是消歧的最优解**：
- 直接针对 WRONG_TARGET 失败模式
- Hard negative 提供最有信息量的梯度信号
- 不需要采样（直接从标注构造），效率极高

---

### 3.4 方法 D：GRPO（对照组）

使用 VLM-R1 标准流程。

```bash
cd VLM-R1/src/open-r1-multimodal
# 配置适配 4090：
# num_generations: 4
# freeze_vision_modules: true
# gradient_checkpointing: true
# use_lora: true
bash run_scripts/run_grpo_rec_lora.sh
```

**超参**：num_generations=4, temperature=0.9, beta=0.04, epsilon=0.2, lr=1e-6

---

### 3.5 方法 E：DAPO（GRPO 的改进版）

在 VLM-R1 GRPO 代码基础上，应用 DAPO 的四个修复：

```python
# train_dapo.py（基于 VLM-R1 修改）

# 改进 1: Clip-Higher — 解耦上下 clip 范围，鼓励探索
epsilon_low = 0.2
epsilon_high = 0.28

# 改进 2: Dynamic Sampling — 过滤方差为 0 的 group
def filter_groups(groups, rewards):
    return [g for g, r in zip(groups, rewards)
            if not all(ri == r[0] for ri in r)]

# 改进 3: Token-level Loss 替代 Sequence-level Loss
# 改进 4: 去掉 KL 惩罚
beta = 0.0
```

---

### 3.6 统一评估框架

```python
METHODS = ["sft", "rft", "online_dpo", "hard_neg_dpo", "grpo", "dapo"]

BENCHMARKS = {
    "in_domain": ["refcoco_val", "refcoco_testA", "refcoco_testB",
                  "refcocop_val", "refcocop_testA", "refcocop_testB",
                  "refcocog_val", "refcocog_test"],
    "out_of_domain": ["lisa_grounding"],
}

METRICS = ["acc@0.5", "acc@0.75", "mean_iou"]

EFFICIENCY = ["gpu_hours", "peak_vram_gb", "samples_per_second"]
```

### 3.7 结果表格模板

**表 1：In-domain 性能（Acc@0.5）**

| Method | RefCOCO val | testA | testB | RefCOCO+ val | testA | testB | RefCOCOg val | test |
|--------|-------------|-------|-------|--------------|-------|-------|--------------|------|
| Zero-shot | | | | | | | | |
| SFT | | | | | | | | |
| RFT | | | | | | | | |
| Online DPO | | | | | | | | |
| Hard Neg DPO | | | | | | | | |
| GRPO | | | | | | | | |
| DAPO | | | | | | | | |
| **Pipeline** | | | | | | | | |

**表 2：Out-of-domain 泛化（LISA Acc@0.5）**

| Method | LISA Acc@0.5 | LISA Mean IoU | Δ vs SFT |
|--------|-------------|--------------|----------|
| SFT | | | baseline |
| RFT | | | |
| Online DPO | | | |
| GRPO | | | |
| DAPO | | | |
| **Pipeline** | | | |

**表 3：效率对比（固定计算预算）**

| Method | GPU Hours | Peak VRAM (GB) | 性能/成本比 |
|--------|-----------|----------------|------------|
| SFT | | | |
| RFT | | | |
| Online DPO | | | |
| GRPO | | | |
| DAPO | | | |

### 3.8 验证检查点

- [ ] 五种方法全部训练完成
- [ ] 所有 benchmark 评估完成
- [ ] 效率指标记录完整
- [ ] 三张对比表格填写完成

---

## Phase 4: 分阶段 Pipeline 设计与验证（Day 41–50）

### 4.1 Pipeline 设计

```
Stage 1: SFT（修复格式 + 建立基础能力）
  ↓
Stage 2: 针对性方法（根据 Phase 1 诊断 + Phase 3 对比结果选择）
  ├── 如果 WRONG_TARGET 是主要残余 → Hard Negative DPO
  ├── 如果 COORDINATE_IMPRECISION 是主要残余 → RFT（高阈值过滤）
  └── 如果 SPATIAL_REASONING 是主要残余 → 启用 <think> + DAPO
  ↓
Stage 3（可选）: DAPO 精调（push the frontier）
```

### 4.2 Pipeline 消融

| 变体 | Stage 1 | Stage 2 | Stage 3 | 目的 |
|------|---------|---------|---------|------|
| Full Pipeline | SFT | DPO/RFT | DAPO | 完整版 |
| 去 Stage 1 | ✗ | DPO/RFT | DAPO | SFT 的必要性 |
| 去 Stage 2 | SFT | ✗ | DAPO | 针对性方法的必要性 |
| 去 Stage 3 | SFT | DPO/RFT | ✗ | RL 精调的必要性 |
| 单一 GRPO | ✗ | ✗ | GRPO only | 对照 |

### 4.3 Thinking 消融

对 DAPO 阶段，对比有无 `<think>` 推理。分开统计简单/复杂 referring expression 的结果。

参考 RadVLM 发现：grounding 上 thinking 不一定有益，但复杂场景可能有帮助。

### 4.4 验证检查点

- [ ] Pipeline 训练完成
- [ ] 消融实验完成（≥4 个变体）
- [ ] Thinking 消融完成
- [ ] Pipeline 优于任何单一方法（或有分析说明为何不优于）

---

## Phase 5: 诊断工具开源（Day 51–55）

### 5.1 工具设计

将 Phase 1 的诊断框架打包成可复用工具——这是区别于 VLM-R1 的独特贡献。

```python
# grounding_diagnose/
# ├── __init__.py
# ├── classify.py       # 失败模式分类器
# ├── visualize.py      # 可视化工具
# ├── report.py         # 自动生成诊断报告
# └── recommend.py      # 基于诊断推荐训练方法

# 使用示例：
from grounding_diagnose import GroundingDiagnoser

diagnoser = GroundingDiagnoser(model, dataset)
report = diagnoser.run()
report.print_summary()
# Output:
# FORMAT_ERROR: 5.2%
# COORDINATE_IMPRECISION: 23.1%
# WRONG_TARGET: 41.3%
# SPATIAL_REASONING_ERROR: 18.7%
# ATTRIBUTE_ERROR: 11.7%
#
# 推荐: Hard Negative DPO → DAPO Pipeline

report.save_visualizations("./vis/")
report.export_markdown("./diagnostic_report.md")
```

---

## Phase 6: 开源发布与技术博客（Day 56–65）

### 6.1 仓库结构

```
VLM-Grounding-PostTraining/
├── README.md
├── grounding_diagnose/        # 诊断工具（独立可复用）
├── methods/                   # 五种训练方法的统一实现
│   ├── sft.py
│   ├── rft.py
│   ├── online_dpo.py
│   ├── hard_neg_dpo.py
│   ├── grpo.py
│   └── dapo.py
├── pipeline/                  # 分阶段 pipeline
├── eval/                      # 评估脚本
├── configs/
├── results/
│   ├── diagnostic_report.md
│   ├── comparison_table.md
│   └── vis/
└── docs/
    └── blog.md
```

### 6.2 博客大纲

```
# 别急着用 GRPO：VLM 视觉 Grounding 的 Post-Training 该怎么选？

1. 为什么大家都在用 GRPO？DeepSeek-R1 带来的路径依赖
2. 第一性原理：GRPO 与 grounding 的结构性不匹配
3. 诊断先行：五种失败模式的分类方法
4. 五种方法的系统对比（同等计算预算）
5. 分阶段 Pipeline 及消融实验
6. 实践建议：单卡 4090 上的最佳策略
```

### 6.3 Hugging Face 发布

上传 pipeline 最终模型 + 各方法中间产物（方便复现对比）。

---

## Phase 7: 面试准备

### 7.1 项目故事线

> "我在海康做生成模型时用过 Flow GRPO 提升识别性能 53%。当我想迁移到 VLM grounding 时，我先停下来问了一个根本问题：GRPO 真的是 grounding 的最优解吗？
>
> 从第一性原理分析，grounding 的输出空间小、reward 连续可计算，GRPO 的核心优势（长链推理 exploration）在这里被削弱。所以我设计了诊断驱动的研究：先对失败模式系统分类，再根据诊断选择方法。
>
> 实验表明 [你的发现]，最终的分阶段 pipeline 在性能和效率上都优于单一 GRPO。"

### 7.2 预期面试问题

| 问题 | 回答要点 |
|------|----------|
| 为什么不直接用 GRPO？ | GRPO 为长链推理设计，grounding 输出短、reward 连续，RFT/DPO 可能更高效 |
| RFT 和 GRPO 的本质区别？ | RFT 只从好样本学，GRPO 还从坏样本的相对排名学。grounding 的连续 reward 让 RFT 的过滤阈值很精确 |
| 诊断框架的价值？ | 把方法选择从"拍脑袋"变成"数据驱动决策"，可复用于其他视觉任务 |
| DAPO 比 GRPO 好在哪？ | Clip-Higher 防 entropy collapse，Dynamic Sampling 过滤无效梯度，去 KL 让学习更充分 |
| 和 VLM-R1 的区别？ | VLM-R1 只证明了 GRPO > SFT；本工作回答了"什么条件下用什么方法"，且组合 > 单一 |

---

## 时间规划总览

| 阶段 | 天数 | 产出 |
|------|------|------|
| Phase 0: 环境与数据 | Day 1–3 | 可运行环境 |
| Phase 1: 诊断分析 | Day 4–10 | 诊断报告 + 失败分类 |
| Phase 2: SFT + 二次诊断 | Day 11–16 | SFT checkpoint + 残余瓶颈分析 |
| Phase 3: 五方法对比 | Day 17–40 | 完整对比表格（性能 + 效率） |
| Phase 4: Pipeline + 消融 | Day 41–50 | 最优 pipeline + 消融证据 |
| Phase 5: 诊断工具 | Day 51–55 | 可复用诊断框架 |
| Phase 6: 发布 | Day 56–65 | GitHub + HF + 博客 |

---

## 成功标准

| 维度 | 最低目标 | 理想目标 |
|------|----------|----------|
| Pipeline vs GRPO（RefCOCO val） | 持平或更高 | 高 ≥2% |
| Pipeline vs GRPO（LISA OOD） | 高 ≥2% | 高 ≥5% |
| Pipeline GPU 时间 vs GRPO | ≤ 同等 | 节省 ≥30% |
| 诊断工具可复用性 | RefCOCO 系列 | 任意 grounding 数据集 |
| GitHub Stars | 100+ | 500+ |
| 面试转化 | 获得 VLM 方向面试 | 获得 offer |

---

## 关键参考

| 资源 | 用途 |
|------|------|
| [VLM-R1](https://github.com/om-ai-lab/VLM-R1) | GRPO 代码基座 |
| [TRL VLM GRPO Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_grpo_trl) | TRL 官方教程 |
| [DAPO](https://arxiv.org/abs/2503.14476) | GRPO 改进方案 |
| [Post-Training in 2026](https://llm-stats.com/blog/research/post-training-techniques-2026) | 方法论背景 |
| [MiMo-VL](https://arxiv.org/pdf/2506.03569) | 多 reward GRPO + grounding |
| [RadVLM + GRPO](https://openreview.net/pdf/1a23300d0c4e0f7a1ebe671fb7c679520fd959c9.pdf) | Grounding + thinking 消融 |
| [LLaVA-Critic-R1](https://github.com/LLaVA-VL/LLaVA-NeXT) | GRPO VLM Critic |
| [Interconnects: GRPO tweaks](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo) | GRPO/DAPO 深度解读 |
| [Vision_GRPO](https://github.com/FusionBrainLab/Vision_GRPO) | 入门参考 |
