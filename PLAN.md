# CLAUDE.md

本文件为 Claude Code 提供在此代码库中工作的指导。

---

## 用户特定指令（务必遵守）

**称呼与交互方式**：
- 每次回答必须以"阿炜"称呼用户
- 执行关键命令或代码时，先解释原因和预期效果，然后让阿炜自己执行
- 所有交流使用中文
- 优先帮助理解代码架构和原理，而非简单执行命令
- 遇到报错时，必须分析根本原因并解释技术原理

---

## 项目背景

**项目名称**：STITCH（Structured Intent Tracking in Contextual History）+ CAME-Bench

**项目定位**：智能体记忆系统，用于对超长多轮对话进行结构化主题标注和意图追踪

**学习目标**：为 AI数据团队-数据策略工程师 岗位面试做准备，重点展示：
1. 工程架构能力 - 理解和跑通复杂开源项目
2. 技术调研能力 - 深入理解智能体记忆、多轮对话技术

**参考资料**：
- 论文：https://arxiv.org/abs/2601.10702
- 网站：https://contextual-intent.github.io
- 数据集：https://huggingface.co/datasets/Seattleyrz/CAME-Bench

---

## 技术栈

- **Python**：3.11.9（虚拟环境 `.venv`）
- **Protocol Buffers**：数据结构定义和序列化
- **Qdrant**：向量数据库（部署在 `http://101.126.29.2:6333`）
- **LLM 提供商**：DashScope（阿里云 qwen-plus 模型，通过 OpenAI 兼容接口）
- **框架**：DSPy（LLM 编排）、litellm（统一 LLM 接口）

---

## 环境配置

### 初始化环境

```bash
# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 生成 Proto 文件（必须先执行）
python scripts/generate_proto_universal.py
```

### 环境变量（`.env` 文件）

```bash
DASHSCOPE_API_KEY=sk-48654e818a354c83b8fe3d5a08e486b7
QDRANT_URL=http://101.126.29.2:6333
```

---

## 关键命令

### 1. 跑通 Benchmark 示例

```bash
python example_run_benchmark.py
```

**说明**：验证环境配置，测试 DashScope API 和数据加载。默认使用占位检索函数，准确率为 0%（预期行为）。

### 2. STITCH 完整 Pipeline

Pipeline 分为 7 个阶段，必须按顺序执行（参考 `scripts/sample_run.sh`）：

```bash
# 阶段 0: 生成 Proto（前置步骤）
python scripts/generate_proto_universal.py

# 阶段 2: 编码对话轮次为向量
python -m method_stitch.dataset_process.encode_turns -c config/encode_config.json

# 阶段 1: 生成数据集描述
python -m method_stitch.dataset_description -c config/dataset_description_config.json

# 阶段 3a: 预测上下文范围
python -m method_stitch.turn_scope_generator \
  --config config/segment_level_note_maintainer_config.json \
  --overwrite

# 阶段 3b: 生成段落级摘要
python -m method_stitch.segment_note_generator \
  --config config/segment_level_note_maintainer_config.json \
  --overwrite

# 阶段 3c: 事件类型标注
python -m method_stitch.event_type_labeler \
  -c config/event_type_labeler_config.json \
  --overwrite

# 阶段 3d: 生成轮次级结构化笔记
python -m method_stitch.turn_level_note_generator \
  -c config/turn_level_note_generator_config.json \
  --overwrite

# 阶段 4: 基于标签的检索
python -m method_stitch.label_based_context_retrieval \
  --config config/label_based_context_retrieval_config.json \
  --overwrite

# 阶段 5: 检索结果格式转换
python -m method_stitch.transform_retrieval_output \
  --config config/transform_retrieval_output_config.json

# 阶段 6: 答案生成
python -m method_stitch.run_answer_generator -c config/answer_generation_config.json

# 阶段 7: 答案评估
python -m method_stitch.run_answer_evaluator -c config/answer_evaluation_config.json
```

---

## STITCH Pipeline 架构

### 核心思想

传统 RAG 直接用向量相似度检索，在超长对话中会检索到大量不相关内容。STITCH 的创新点是**先进行结构化标注，再基于标签过滤+向量检索**。

### 数据流

```
输入: turns.jsonl (对话轮次) + questions.jsonl (评估问题)
  │
  ├─ 阶段 2: 向量编码 → Qdrant 向量库
  ├─ 阶段 1: 数据集描述生成 → 功能细节种子
  │
  ├─ 阶段 3: Intent Tracking（STITCH 核心）
  │   ├─ 3a. 预测上下文范围 (context_scope)
  │   ├─ 3b. 段落级摘要 (segment_level_notes)
  │   ├─ 3c. 事件类型标注 (event_types)
  │   └─ 3d. 轮次级结构化笔记 (act, target, scope, events)
  │
  ├─ 阶段 4: 标签检索
  │   └─ 问题 → LLM 选择标签 → 过滤候选轮次 → 向量排序 → top-k
  │
  └─ 阶段 5-7: 格式转换 → 答案生成 → 评估
```

### 关键数据结构

**turns.jsonl** - 对话轮次：
```json
{
  "id": "turn_1",
  "role": "user",
  "content": "I'm looking for a hotel in Tokyo.",
  "partition": ["conv_1"],
  "timestamp_mapping": {"conv_1": "2023-10-01T10:00:00Z"}
}
```

**questions.jsonl** - 评估问题：
```json
{
  "id": "q_1",
  "content": "What was the user's budget?",
  "type": "FREE_FORM",
  "answer": {"free_form_answer": "Under $200 per night"},
  "question_turn_ids": ["turn_5"],
  "answer_turn_ids": ["turn_2", "turn_3"]
}
```

---

## 数据集说明

CAME-Bench 提供 14 个 trajectories，分为 3 种规模：

| Scale  | Trajectories       | Turn Range   | 用途                |
|--------|-------------------|--------------|-------------------|
| Small  | traj-8 至 traj-13 | 62-240 turns | 快速测试（推荐新手） |
| Medium | traj-2 至 traj-7  | 312-1442     | 平衡测试            |
| Large  | traj-0, traj-1    | 912-4105     | 完整评估            |

**成本估算（基于 DashScope qwen-plus）**：
- traj-0 (912 turns)：约 ¥0.40 RMB
- traj-8 (62 turns)：约 ¥0.05 RMB

**当前进度**：正在处理 traj-0（Large 规模）

---

## 已修复的问题

### 1. Unicode 编码问题（Windows GBK 默认编码）

**症状**：`UnicodeDecodeError: 'gbk' codec can't decode byte...`

**修复位置**：
- `came_bench/utils/io.py:156, 158, 168, 170` - 所有 `open()` 调用添加 `encoding="utf-8"`

### 2. Proto 导入路径错误

**症状**：`ModuleNotFoundError: No module named 'src'`

**修复位置**：
- `method_stitch/dataset_process/encode_turns.py:28-32` - 改为 `from came_bench.proto import`

### 3. load_turns() 函数签名不匹配

**症状**：`TypeError: load_turns() takes 1 positional argument but 2 were given`

**原因**：多个模块仍在调用旧版 API `load_turns(dataset_name, turns_path)`，但新版只接受 1 个参数

**修复位置**：
- `method_stitch/dataset_process/encode_turns.py:137`
- `method_stitch/turn_scope_generator.py:101`
- `method_stitch/segment_note_generator.py:137`
- `method_stitch/event_type_labeler.py:1042`

全部改为：`load_turns(turns_path)`

### 4. DashScope API 限流

**症状**：`429 Too Many Requests`

**修复**：`config/encode_config.json` 中 `max_concurrent` 从 16 降至 4

---

## 配置文件说明

所有配置文件位于 `config/` 目录，基于 `sample_config_files/` 模板创建：

| 配置文件                                  | 用途            | 关键参数                          |
|----------------------------------------|---------------|-------------------------------|
| `encode_config.json`                   | 阶段 2 向量编码    | max_concurrent, vector_size    |
| `dataset_description_config.json`      | 阶段 1 数据集描述   | sample_rate                    |
| `segment_level_note_maintainer_config.json` | 阶段 3a/3b     | scope_history_window           |
| `event_type_labeler_config.json`       | 阶段 3c 事件标注   | -                              |
| `turn_level_note_generator_config.json`| 阶段 3d 笔记生成   | -                              |
| `label_based_context_retrieval_config.json` | 阶段 4 检索   | top_k                          |
| `transform_retrieval_output_config.json`| 阶段 5 格式转换   | -                              |
| `answer_generation_config.json`        | 阶段 6 答案生成    | max_tokens                     |
| `answer_evaluation_config.json`        | 阶段 7 答案评估    | -                              |

**通用 LLM 配置**（所有阶段共享）：
```json
{
  "provider": "LANGUAGE_MODEL_PROVIDER_OPENAI",
  "model_name": "openai/qwen-plus",
  "temperature": 1.0,
  "max_tokens": 4096,
  "openai_config": {
    "api_key": "${DASHSCOPE_API_KEY}",
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1"
  }
}
```

---

## 目录结构

```
contextual-intent/
├── came_bench/                # Benchmark 框架
│   ├── benchmark.py           # Benchmark 主类
│   ├── data_process/          # 数据下载和解码
│   ├── pipeline/              # 答案生成和评估
│   └── utils/                 # 工具函数（lm, io, encoder）
│
├── method_stitch/             # STITCH 算法实现
│   ├── dataset_description.py
│   ├── dataset_process/encode_turns.py
│   ├── turn_scope_generator.py
│   ├── segment_note_generator.py
│   ├── event_type_labeler.py
│   ├── turn_level_note_generator.py
│   ├── label_based_context_retrieval.py
│   └── segment_level_note_maintainer.py  # 核心类
│
├── proto/                     # Protobuf 定义
│   ├── project_dataset_uniform.proto
│   ├── language_model_provider.proto
│   └── context_reduction_retrieval.proto
│
├── generated_proto/           # 编译后的 Python 类
│
├── config/                    # 配置文件（自定义）
├── stitch_output/             # Pipeline 输出
├── came_bench_data/decoded/   # 解码后的数据集
│
├── scripts/
│   ├── sample_run.sh          # 完整 pipeline 脚本
│   └── generate_proto_universal.py
│
├── example_run_benchmark.py   # Benchmark 示例入口
├── requirements.txt
└── .env                       # 环境变量
```

---

## 当前工作进度

**已完成阶段**：
- ✅ 阶段 0: Proto 生成
- ✅ 阶段 2: 向量编码（912 turns → Qdrant，耗时约 10 分钟，成本 ¥0.003）
- ✅ 阶段 1: 数据集描述（生成 3.6KB 描述文件）
- ✅ 阶段 3a: 上下文范围预测（150 turns 测试运行）
- ✅ 阶段 3b: 段落级摘要（108 segments，251KB）

**进行中**：
- 🔄 阶段 3c: 事件类型标注

**待完成**：
- ⏳ 阶段 3d: 轮次级结构化笔记
- ⏳ 阶段 4: 基于标签的检索
- ⏳ 阶段 5-7: 格式转换 + 答案生成 + 评估

**待决策**：
- 是否继续处理 traj-0 全部 912 turns（成本 ~¥0.40）
- 还是切换到 Small 数据集 traj-8（成本 ~¥0.05）进行快速验证

---

## 常见问题 FAQ

### Q1: 为什么使用 Protocol Buffers？

**原因**：项目需要在多个模块间传递复杂数据结构，protobuf 提供类型安全、跨语言兼容和高效序列化。

### Q2: 为什么 Pipeline 分这么多阶段？

**原因**：每个阶段产生中间产物可独立复用，便于调试和增量处理。例如阶段 2 的向量编码只需运行一次，后续阶段可重复使用。

### Q3: STITCH 相比传统 RAG 的优势？

**传统 RAG**：问题 → 向量检索 → 返回 top-k

**STITCH**：问题 → LLM 预测标签 → 标签过滤候选集 → 向量检索 → 返回 top-k

**优势**：在超长对话（1000+ 轮）中，标签过滤能将候选集从 1000 降至 50-100，显著提升检索精度和召回率。

### Q4: 为什么阶段 3a/3b 只处理了 150/912 turns？

**原因**：未明确，可能是：
1. 中途遇到 API 限流错误，触发 checkpoint 保存后提前退出
2. 配置中存在未发现的采样参数
3. 代码逻辑中存在早停条件

**建议**：切换到 Small 数据集（traj-8，62 turns）进行完整验证，再处理大数据集。

---

## 技术调研方向

当项目跑通后，需要深入调研以下方向以支撑面试：

1. **多轮对话生成技术**
   - 业界主流方法对比（STITCH vs MemGPT vs Letta）
   - 长期记忆管理策略

2. **意图追踪与主题分割**
   - 对话主题边界检测算法
   - 上下文范围预测方法

3. **检索增强生成（RAG）**
   - Dense retrieval vs Sparse retrieval
   - 混合检索策略
   - 标签辅助检索的学术进展

4. **智能体记忆系统**
   - 短期记忆 vs 长期记忆
   - 记忆整合与遗忘机制
   - 分层记忆架构

---

## 调试技巧

### 查看 Qdrant 向量库内容

```bash
curl http://101.126.29.2:6333/collections/traj-0
```

### 查看中间产物

```bash
# 数据集描述
cat stitch_output/traj-0/dataset_description.txt

# 上下文范围分配
cat stitch_output/traj-0/context_scope_assignments.json | jq

# 段落级摘要
cat stitch_output/traj-0/segment_level_notes.jsonl | jq -s
```

### 估算成本

每个阶段的 API 调用次数约等于对话轮次数量（N turns），qwen-plus 价格：
- 输入：¥0.004 / 1k tokens
- 输出：¥0.012 / 1k tokens

估算公式：`成本 ≈ N * 平均 tokens * 价格`

---

## 重要提醒

1. **先理解再执行**：每个阶段执行前，必须理解其输入输出和作用
2. **检查中间产物**：每个阶段完成后，检查生成的文件内容是否符合预期
3. **成本控制**：优先用 Small 数据集验证完整 pipeline，再处理大数据集
4. **保留现场**：遇到错误时，保留日志和中间文件，便于分析根因
5. **遵循指令**：严格遵守本文件开头的"用户特定指令"

---

**最后更新**：2026-02-13
**当前状态**：阶段 3c 进行中
