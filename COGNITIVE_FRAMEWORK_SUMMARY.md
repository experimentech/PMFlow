# Cognitive Architecture Framework - Added to PMFlow v0.3.6

## Summary

Added a complete cognitive architecture framework to PMFlow that composes existing primitives into a high-level agent system.

## What Was Added

### 1. Core Module (`pmflow/cognitive.py`)
- **CognitiveAgent**: Main agent class that orchestrates reasoning
- **WorkingMemory**: Short-term memory with relevance decay
- **MetacognitiveMonitor**: Analyzes reasoning processes for issues
- **ReasoningMode**: Enum for different reasoning strategies
- **ReasoningTrace**: Complete record of a reasoning process
- **MemoryItem**: Individual memory with timestamp and relevance

### 2. Reasoning Modes
- **DIRECT**: Single-step encoding (fast)
- **ITERATIVE**: Multi-step trajectory tracing (observable)
- **GOAL_DIRECTED**: Intent-biased reasoning (controllable)
- **EXPLORATORY**: Try multiple paths (robust)

### 3. Key Features
- **Observable**: Full visibility into reasoning trajectories
- **Explainable**: Human-readable reasoning explanations
- **Controllable**: Set goals, mark hazards, guide reasoning
- **Adaptive**: Metacognitive monitoring with suggestions
- **Compositional**: Reuses all existing PMFlow primitives

### 4. Documentation
- `docs/cognitive_architecture.md`: Complete guide with examples
- `examples/cognitive_agent_demo.py`: Working demonstration
- Updated `pmflow/__init__.py`: All exports properly configured

## Architecture Design

```
CognitiveAgent
├── Perception (PMFlowEmbeddingEncoder)
├── Working Memory (relevance-weighted recall)
├── Reasoning (4 modes using PMFlow trajectories)
└── Metacognition (process monitoring & feedback)
```

## Why This Is Valuable

### 1. Fills a Gap
PMFlow had all the low-level primitives (trajectories, intent, hazards) but no high-level framework for building actual cognitive systems.

### 2. Genuinely Novel
No other architecture lets you:
- Observe the actual semantic path of reasoning
- Dynamically steer reasoning with physics-based intent
- Monitor and explain cognitive processes
- Compose multiple reasoning strategies

### 3. Practical
Not just theoretical - provides working implementations of:
- Memory management
- Goal-directed reasoning
- Explainability
- Metacognitive monitoring

### 4. Extensible
Clean abstractions make it easy to add:
- Long-term memory
- Multi-agent systems
- Learning from traces
- Custom reasoning modes

## Example Usage

```python
from pmflow import PMFlowEmbeddingEncoder, CognitiveAgent, ReasoningMode

# Create agent
encoder = PMFlowEmbeddingEncoder(dimension=96, enable_flow=True)
agent = CognitiveAgent(encoder, enable_metacognition=True)

# Set a goal
agent.set_goal(["accuracy"], strength=0.7)

# Reason about something
result, trace = agent.think(
    ["machine", "learning"],
    mode=ReasoningMode.GOAL_DIRECTED
)

# Get explanation
print(agent.explain_reasoning(trace))
# Shows: path length, efficiency, concepts visited, metacognitive insights
```

## Integration with Existing PMFlow

Reuses all existing features:
- `PMFlowEmbeddingEncoder.trace_trajectory()` → reasoning traces
- `PMFlowEmbeddingEncoder.inject_intent()` → goal-directed mode
- `ParallelPMField.mark_as_hazard()` → hazard avoidance
- `ParallelPMField.mark_as_attractor()` → goal reinforcement
- Multi-scale fields → hierarchical reasoning

No breaking changes - purely additive.

## Testing

```bash
# Test import
python -c "from pmflow import CognitiveAgent; print('ok')"

# Run demo
python examples/cognitive_agent_demo.py

# Outputs:
# ✓ Multiple reasoning modes
# ✓ Goal-directed thinking
# ✓ Working memory with context
# ✓ Metacognitive monitoring
# ✓ Explainable reasoning traces
```

## Files Modified

1. `pmflow/cognitive.py` - New module (500+ lines)
2. `pmflow/__init__.py` - Added exports and version bump to 0.3.6
3. `docs/cognitive_architecture.md` - Complete documentation
4. `examples/cognitive_agent_demo.py` - Working demonstration

## Version

Updated from `0.3.5` → `0.3.6`

## Next Steps (Optional)

Potential future extensions:
- Long-term episodic memory
- Learning from reasoning traces
- Multi-agent coordination
- Integration with PMFlowLanguageModel
- Curriculum learning using trajectory metrics

## Conclusion

This framework makes PMFlow's unique capabilities (physics-based reasoning, observable trajectories, controllable flow) accessible at a high level for building practical cognitive systems. It's the "missing piece" that turns PMFlow from a collection of primitives into a complete cognitive architecture.
