## [EMNLP2026] LifeMem: Enabling Lifelong Experience Reuse for LLM Agents ##

我们提出了基于技能聚类的Agent终身学习框架，通过从训练轨迹中自主提取、组织和检索技能知识，Agent能够在不同环境中复用可迁移的经验，同时减少灾难性遗忘。

## 系统概述

该系统通过三个阶段构建Agent的长期记忆：
1. **建立阶段**：从训练轨迹中识别和分类任务模式，形成初始记忆簇
2. **整理阶段**：对记忆进行去重和清理，移除冗余信息
3. **完善阶段**：从记忆簇中提取和总结技能文档

建立完成后，系统提供HTTP检索接口，可集成到Agent工作流中。

## 快速开始

### Memory构建流程

#### 阶段1：建立记忆

从训练轨迹文件中分类任务，构建初始记忆簇：

```bash
python ./memory/build_memory.py --stage 1
```

**配置项**（在 [build_memory.py](build_memory.py) 中修改）：
- `TRAIN_FILE`: 训练轨迹文件路径（默认：`train_alfworld.jsonl`）
- `TASK`: 任务名称（默认：`alfworld_mrl`）
- `first_time`: 首次运行设为 `1`，后续增量更新设为 `2`

**输出**：
- `memory_faiss_<TASK>/`: FAISS向量索引
- `memory_cluster_<TASK>.db`: SQLite聚类数据库

#### 阶段2：整理记忆

对记忆簇进行去重，移除冗余轨迹：

```bash
python ./memory/build_memory.py --stage 2
```

该阶段会检测重复项并重建FAISS索引。

#### 阶段3：完善记忆

从记忆簇中总结提取技能表示（representations/insights）：

```bash
python ./memory/build_memory.py --stage 3
```

生成的技能文档将存储在SQLite数据库的`clusters`表中。

### 启动检索服务

完成Memory构建后，启动HTTP API服务：

```bash
python ./memory/server.py
```

**默认配置**：
- 主机：`0.0.0.0`（监听所有接口）
- 端口：`7000`
- 任务：`alfworld_mrl`

**配置修改**：编辑 [server.py](server.py) 中的以下变量：
```python
TASK_db = "your_task_name"
TASK_faiss = "your_task_name"
HOST = "127.0.0.1"  # 仅本地访问
PORT = 7000
```

## 训练数据
从以下仓库获取这些数据集：
- AlfWorld：**[Aligning Text and Embodied Environments for Interactive Learning](https://github.com/alfworld/alfworld#quickstart)**
- ScienceWorld：**[ScienceWorld: Is your Textual Agent Smarter than a 5th grader?](https://github.com/allenai/ScienceWorld)**
- WebShop：**[WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://github.com/princeton-nlp/WebShop/tree/master)**
- Mind2Web, Miniwob++：**[Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control](https://github.com/ltzheng/Synapse)**
- 我们通过基于GPT-4.1采样并过滤的方式，为其他任务构建了2k+条的高质量训练集，新的轨迹数据存储在`/trainset`目录
