# PMFlow Cognitive Architecture

A composable framework for building cognitive agents using PMFlow's physics-based reasoning primitives.

## Overview

The cognitive architecture provides high-level abstractions for:
- **Perception**: Encoding inputs into semantic space
- **Memory**: Short-term working memory with relevance decay
- **Reasoning**: Multiple modes (direct, iterative, goal-directed, exploratory)
- **Metacognition**: Monitoring and analyzing reasoning processes

## Quick Start

```python
from pmflow import PMFlowEmbeddingEncoder, CognitiveAgent, ReasoningMode

# Create encoder with flow enabled
encoder = PMFlowEmbeddingEncoder(
    dimension=96,
    latent_dim=48,
    enable_flow=True
)

# Create cognitive agent
agent = CognitiveAgent(
    encoder=encoder,
    memory_capacity=10,
    enable_metacognition=True
)

# Reason about something
result, trace = agent.think(
    ["machine", "learning"],
    mode=ReasoningMode.ITERATIVE
)

# Get explanation
print(agent.explain_reasoning(trace))
```

## Reasoning Modes

### Direct (Single-Step)
```python
result, trace = agent.think(query, mode=ReasoningMode.DIRECT)
```
Simple one-shot encoding. Fast, no trajectory tracking.

### Iterative (Multi-Step)
```python
result, trace = agent.think(query, mode=ReasoningMode.ITERATIVE)
```
Traces a trajectory through semantic space. Shows the "path" of reasoning.

### Goal-Directed
```python
agent.set_goal(["neural", "networks"], strength=0.7)
result, trace = agent.think(query, mode=ReasoningMode.GOAL_DIRECTED)
```
Biases reasoning toward goal concepts using frame-dragging physics.

### Exploratory
```python
result, trace = agent.think(query, mode=ReasoningMode.EXPLORATORY, max_iterations=5)
```
Tries multiple reasoning paths, picks the most efficient one.

## Working Memory

The agent maintains short-term memory of recent reasoning:

```python
# Memory is automatically populated during thinking
agent.think(["machine", "learning"])

# Recall relevant memories
query_embedding = encoder.encode(["AI"])
recalled_memories = agent.memory.recall(query_embedding, top_k=3)

for memory in recalled_memories:
    print(f"Trace: {memory.trace.query}")
    print(f"Relevance: {memory.relevance}")
```

Memory items decay over time and older/less-relevant items are forgotten.

## Metacognition

The metacognitive monitor analyzes reasoning traces:

```python
agent = CognitiveAgent(encoder, enable_metacognition=True)
result, trace = agent.think(query)

# Check metacognitive analysis
if 'metacognition' in trace.metadata:
    analysis = trace.metadata['metacognition']
    
    if analysis['high_effort']:
        print("Reasoning required significant mental effort")
    
    if analysis['stuck']:
        print("Got stuck in circular reasoning")
    
    for suggestion in analysis['suggestions']:
        print(f"Suggestion: {suggestion}")
```

The monitor tracks:
- **High effort**: Long reasoning paths
- **Inefficiency**: Meandering trajectories
- **Getting stuck**: Circular patterns
- **Convergence**: Successfully reaching goal

## Reasoning Traces

Every reasoning operation returns a trace:

```python
@dataclass
class ReasoningTrace:
    query: List[str]                # Original query tokens
    trajectory: torch.Tensor        # Path through semantic space
    metrics: Dict[str, float]       # Path length, efficiency, etc.
    attractors_visited: List[int]   # Which centers influenced reasoning
    hazards_avoided: List[int]      # Which hazards were avoided
    mode: ReasoningMode             # How it reasoned
    duration: float                 # Wall-clock time
    success: bool                   # Did it succeed?
    metadata: Dict[str, Any]        # Metacognition, etc.
```

Use traces for:
- Debugging reasoning failures
- Explaining decisions
- Learning from experience
- Optimizing future reasoning

## Advanced Features

### Goal Management
```python
# Set multiple goals
agent.set_goal(["accuracy"], strength=0.6)
agent.set_goal(["efficiency"], strength=0.4)

# Clear goals
agent.clear_goals()
```

### Hazard Marking
```python
# Mark regions to avoid
bad_location = encoder.encode(["unsafe", "concept"])
agent.mark_hazard(bad_location, radius=1.0)
```

### Explanation Generation
```python
result, trace = agent.think(query)
explanation = agent.explain_reasoning(trace)
print(explanation)
# Output:
# Reasoning Mode: iterative
# Query: machine learning
# Path Length (Mental Effort): 0.234
# Efficiency: 0.895
# Semantic Concepts Visited: [3, 7, 12]
```

## Integration with PMFlow Features

The cognitive architecture leverages all PMFlow capabilities:

- **Trajectory tracing** (`trace_trajectory`)
- **Intent injection** (`inject_intent`)
- **Center manipulation** (`mark_as_hazard`, `mark_as_attractor`)
- **Multi-scale fields** (hierarchical reasoning)
- **BioNN components** (uncertainty-aware reasoning)

## Use Cases

### 1. Explainable AI
Track and explain how the agent reached a decision by analyzing semantic trajectories.

### 2. Interactive Agents
Agents that adapt their reasoning based on goals and constraints during conversation.

### 3. Research Tools
Study cognitive processes by observing reasoning traces and metacognitive patterns.

### 4. Adaptive Systems
Systems that learn from their own reasoning patterns using metacognitive feedback.

## Design Philosophy

1. **Composable**: Mix and match components as needed
2. **Observable**: Full visibility into reasoning processes
3. **Controllable**: Set goals, mark hazards, guide reasoning
4. **Practical**: Solves real problems, not just theoretical exercises

## Examples

See `examples/cognitive_agent_demo.py` for a complete demonstration.

## Requirements

- PMFlow >= 0.3.6
- `enable_flow=True` for iterative, goal-directed, and exploratory reasoning
- Working memory requires keeping encoder in memory

## Future Directions

Potential extensions:
- Long-term memory (episodic/semantic)
- Multi-agent coordination
- Learning from reasoning traces
- Adaptive meta-strategies
- Integration with language models
