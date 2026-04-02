"""
Cognitive Agent Example

Demonstrates using PMFlow's cognitive architecture framework to build
an agent that can reason, remember, and adapt.
"""

from pmflow import PMFlowEmbeddingEncoder, CognitiveAgent, ReasoningMode


def main():
    print("=" * 70)
    print("PMFlow Cognitive Architecture Demo")
    print("=" * 70)
    
    # Create encoder with flow enabled
    print("\n1. Creating cognitive agent...")
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=48,
        enable_flow=True,
        seed=42
    )
    
    agent = CognitiveAgent(
        encoder=encoder,
        memory_capacity=10,
        enable_metacognition=True
    )
    print("   ✓ Agent created with working memory and metacognition")
    
    # Example 1: Simple direct reasoning
    print("\n2. Direct reasoning (single-step)...")
    result, trace = agent.think(
        ["machine", "learning"],
        mode=ReasoningMode.DIRECT
    )
    print(agent.explain_reasoning(trace))
    
    # Example 2: Iterative reasoning (multi-step)
    print("\n3. Iterative reasoning (trace through semantic space)...")
    result, trace = agent.think(
        ["artificial", "intelligence"],
        mode=ReasoningMode.ITERATIVE
    )
    print(agent.explain_reasoning(trace))
    
    # Example 3: Goal-directed reasoning
    print("\n4. Goal-directed reasoning (with intent)...")
    agent.set_goal(["neural", "networks"], strength=0.7)
    result, trace = agent.think(
        ["deep", "learning"],
        mode=ReasoningMode.GOAL_DIRECTED
    )
    print(agent.explain_reasoning(trace))
    agent.clear_goals()
    
    # Example 4: Memory recall
    print("\n5. Working memory demonstration...")
    print(f"   Memory contains {len(agent.memory.items)} items")
    
    # Try to recall something related to earlier queries
    query_embedding = encoder.encode(["learning"])
    recalled = agent.memory.recall(query_embedding, top_k=2)
    print(f"   Recalled {len(recalled)} relevant memories")
    for i, mem in enumerate(recalled):
        if mem.trace:
            print(f"   Memory {i+1}: {' '.join(mem.trace.query)} (relevance: {mem.relevance:.3f})")
    
    # Example 5: Metacognitive analysis
    print("\n6. Metacognitive insights...")
    if agent.metacognition and agent.metacognition.history:
        latest_trace = agent.metacognition.history[-1]
        analysis = latest_trace.metadata.get('metacognition')
        if analysis:
            print(f"   High effort: {analysis['high_effort']}")
            print(f"   Inefficient: {analysis['inefficient']}")
            print(f"   Stuck: {analysis['stuck']}")
            if analysis['suggestions']:
                print("   Suggestions:")
                for sug in analysis['suggestions']:
                    print(f"     - {sug}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    
    print("\nKey Features Demonstrated:")
    print("  ✓ Multiple reasoning modes")
    print("  ✓ Goal-directed thinking")
    print("  ✓ Working memory with context")
    print("  ✓ Metacognitive monitoring")
    print("  ✓ Explainable reasoning traces")


if __name__ == "__main__":
    main()
