cluster_trajectory = """
Role & Mission
You are an advanced reasoning agent responsible for organizing task trajectories into skill-based clusters.

You are operating in incremental assignment mode. Your role is to assign new trajectories to existing clusters or to create a new cluster if necessary.

You must NOT delete, merge, or split existing clusters in this mode.

Each trajectory corresponds to a task execution, and tasks may require overlapping or distinct underlying skills (e.g., planning, tool use, symbolic reasoning, perception, long-horizon decomposition).

Input Data
New Trajectories: {trajectories}

You also have access to an existing memory cluster library.
Below are the EXISTING CLUSTERS retrieved as the most relevant candidates for comparison. This is a partial view of the memory, not an exhaustive list of all clusters. Each cluster represents a group of trajectories that rely on similar core skills: 
{existing_clusters}

Your goal is to group trajectories according to the similarity of the skills required to solve the tasks, focusing on what capabilities are exercised rather than superficial task details or domains. Solve the problem with the following steps:

Step 1: Analysis (Chain of Thought)
Before outputting the final operations, perform a systematic analysis for each trajectory:

Deconstruct Skills: Identify the core reasoning requirements of the trajectory
Evaluate Overlap: Compare these skills against the existing_clusters.

Determine Operation: - ADD: If the trajectory's skills overlap significantly or logically fit within the scope of an existing cluster. Prioritize broadening an existing cluster over creating a new one.

NEW: Only if the required skills represent a completely different domain with no meaningful overlap.

Step 2: Final Output (Operations)
Based on your analysis, output the actions using this operations:
ADD <TRAJECTORY_ID> to <CLUSTER_NAME>
NEW <CLUSTER_NAME> with <TRAJECTORY_ID>

Response Format:
Please format your response as follows:
<Analysis>
[Provide a brief, step-by-step reasoning process here. Identify the 3-6 sub-skills found in the trajectories and explain why certain IDs belong together.]
<Execution>
NEW <Cluster_A> with <ID_1>
ADD <ID_3> to <Cluster_A>

Guidelines & Constraints:
Format: <CLUSTER_NAME> must use underscores (_) instead of spaces. TRAJECTORY_ID must match the 'id' field.
Assignment: A trajectory may belong to only one cluster. You can only add one trajectory per operation line.
Skill Focus: Base decisions on core skills and reasoning requirements, not task topics. Do not invent unrelated skills.
Cluster Creation: Create a NEW cluster only when a clear and defensible skill gap exists. Descriptions should focus on reasoning type and primary capabilities."""

get_summary = """You are an advanced reasoning agent responsible for organizing task trajectories into skill-based clusters.

Here are a few trajectories in one cluster, your task is to summarize these trajectories into one sentence, which is a description of the cluster, like their shared core skills, or the core tasks they can solve.

{trajectories}

Here is a previous summary which may help you:
{summary}

Output format:
Your summary should begin with "This cluster is about ..."
"""

get_split="""## Role
You are an advanced reasoning agent specialized in deconstructing and refining skill-based memory clusters. You operate in **CLUSTER SPLIT** mode to ensure high-precision skill categorization.

## Context
A memory cluster has grown too large (>50 trajectories), indicating it now covers too many distinct skills. You must dissolve the old cluster and redistribute **ALL** trajectories into 3-6 new, more granular clusters.

- **Old Cluster Name:** {cluster_id}
- **Old Cluster Summary:** {summary}

## Input: Trajectories to Redistribute
{examples}

## Task Instructions
1.  Deconstruction Analysis (CoT): First, analyze the provided trajectories. Identify the underlying granular reasoning patterns (e.g., 'Temporal_Logic_Reasoning', 'Nested_Tool_Coordination'). Group them mentally based on structural similarity rather than broad topics.
2.  Assignment: Assign EVERY trajectory ID from the input list to exactly one new cluster.
3.  Execution: Output the reorganization using the specified operational commands.

## Operational Commands
- NEW <CLUSTER_NAME> with <TRAJECTORY_ID>: Create a new category using a seed trajectory.
- ADD <TRAJECTORY_ID> to <CLUSTER_NAME>: Assign additional trajectories to an existing new category.

## Constraints
- Total Coverage: Every Trajectory ID must appear exactly once.
- Naming: Use `Snake_Case_Naming`. Do not reuse the old cluster names: `{old_cluster_id}`.
- Granularity: Clusters should reflect specific logical workflows or technical patterns.
- Output Order: You must list all NEW operations first, followed by all ADD operations.

---

## Response Format
Please format your response as follows:
<Analysis>
[Provide a brief, step-by-step reasoning process here. Identify the 3-6 sub-skills found in the trajectories and explain why certain IDs belong together.]

<Execution>
NEW <Cluster_A> with <ID_1>
NEW <Cluster_B> with <ID_2>
...
ADD <ID_3> to <Cluster_A>
ADD <ID_4> to <Cluster_B>
"""

get_representations="""You are an advanced reasoning agent that can generate a concise rule set based on forming new insights from past task trajectories. You will be given successful trials.

Here are the trials: {success_history}

By examining the successful trials, you will extract 3 to 5 rules that are general and high-level insights of the successful trajectories or proposed Way of Thought. These rules should serve as helpful tips for different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action.

Follow the below format: 
RULE 1: rule content... 
RULE 2: rule content...

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Generate at least 3 and at most 5 rules in total.
"""