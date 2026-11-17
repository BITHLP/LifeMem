# LifeMem: Enabling Lifelong Experience Reuse for LLM Agents

We propose a lifelong learning framework for LLM agents based on skill clustering. The framework autonomously extracts, organizes, and retrieves skill knowledge from training trajectories. This lets agents reuse transferable experience across environments while reducing catastrophic forgetting.

## Overview

The system builds an agent's long-term memory in three stages:

1. **Build**: identify and classify task patterns in the training trajectories to form initial memory clusters.
2. **Consolidate**: deduplicate and clean the memory to remove redundant information.
3. **Refine**: extract and summarize skill documents from the memory clusters.

Once the memory is built, the system provides an HTTP retrieval API that can be integrated into agent workflows.

## Quick Start

### Building the Memory

#### Stage 1: Build

Classify tasks from the training trajectory file and construct the initial memory clusters:

```bash
python ./memory/build_memory.py --stage 1
```

**Configuration** (edit in [build_memory.py](build_memory.py)):

- `TRAIN_FILE`: path to the training trajectory file (default: `train_alfworld.jsonl`)
- `TASK`: task name (default: `alfworld_mrl`)
- `first_time`: set to `1` for the first run and `2` for later incremental updates

**Outputs**:

- `memory_faiss_<TASK>/`: FAISS vector index
- `memory_cluster_<TASK>.db`: SQLite cluster database

#### Stage 2: Consolidate

Deduplicate the memory clusters and remove redundant trajectories:

```bash
python ./memory/build_memory.py --stage 2
```

This stage detects duplicates and rebuilds the FAISS index.

#### Stage 3: Refine

Summarize and extract skill representations (representations/insights) from the memory clusters:

```bash
python ./memory/build_memory.py --stage 3
```

The generated skill documents are stored in the `clusters` table of the SQLite database.

### Starting the Retrieval Service

After the memory is built, start the HTTP API service:

```bash
python ./memory/server.py
```

**Defaults**:

- Host: `0.0.0.0` (listens on all interfaces)
- Port: `7000`
- Task: `alfworld_mrl`

**Changing the configuration**: edit the following variables in [server.py](server.py):

```python
TASK_db = "your_task_name"
TASK_faiss = "your_task_name"
HOST = "127.0.0.1"  # local access only
PORT = 7000
```

### Evaluation

#### ALFWorld, HotpotQA, WebShop

See the evaluation scripts and testset in the **[ExpeL](https://github.com/LeapLabTHU/ExpeL)** repository.

#### MiniWoB++, Mind2Web

See the evaluation scripts and testset in the **[Synapse](https://github.com/ltzheng/Synapse)** repository.

#### tau-bench

See the evaluation scripts and testset in the official **[tau-bench](https://github.com/sierra-research/tau-bench)** repository.

#### ScienceWorld

The evaluation is based on the official **[ScienceWorld](https://github.com/allenai/ScienceWorld)** repository.

Replace the `examples` directory of the official repository with the files in this project's `./ScienceWorld/examples` (which includes our automatic agent evaluation script `evaluator.py`), then run:

```bash
python evaluator.py --model_name "qwen3-32b" --output_file "test_qwen3-32b.jsonl"
```

#### API-Bank

Run the following command to start the evaluation:

```bash
python evaluator.py --model "gpt-4o-mini" --output_file "test-4omini.json"
```

#### ToolQA

We use the Tabular Database task. Run:

```bash
python ./benchmark/ReAct/code/tests.py
```

## Training Data

Obtain the datasets from the following repositories:

- ALFWorld: **[Aligning Text and Embodied Environments for Interactive Learning](https://github.com/alfworld/alfworld#quickstart)**
- ScienceWorld: **[ScienceWorld: Is your Textual Agent Smarter than a 5th Grader?](https://github.com/allenai/ScienceWorld)**
- WebShop: **[WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents](https://github.com/princeton-nlp/WebShop/tree/master)**
- Mind2Web, MiniWoB++: **[Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control](https://github.com/ltzheng/Synapse)**

For the other tasks, we built high-quality training sets of 2k+ trajectories by sampling with GPT-4.1 and filtering the results. These new trajectories are stored in the `/trainset` directory.
