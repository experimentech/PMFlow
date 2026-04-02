"""
PMFlow Cognitive Architecture Framework

A composable framework for building cognitive agents using PMFlow's physics-based
reasoning primitives. Provides high-level abstractions for perception, memory,
reasoning, and metacognition.

Design Principles:
1. Modular - mix and match components
2. Observable - track reasoning traces
3. Controllable - inject goals/constraints at runtime
4. Practical - solves real problems, not just demos
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time

from pmflow import PMFlowEmbeddingEncoder
from pmflow.core.pmflow import ParallelPMField


class ReasoningMode(Enum):
    """How the agent reasons through a problem"""
    DIRECT = "direct"              # Single-step encoding
    ITERATIVE = "iterative"        # Multi-step refinement
    GOAL_DIRECTED = "goal_directed"  # With intent injection
    EXPLORATORY = "exploratory"    # With exploration bonus


@dataclass
class ReasoningTrace:
    """Record of a reasoning process"""
    query: List[str]
    trajectory: torch.Tensor
    metrics: Dict[str, float]
    attractors_visited: List[int]
    hazards_avoided: List[int]
    mode: ReasoningMode
    duration: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryItem:
    """Item in working memory"""
    content: torch.Tensor  # Latent representation
    timestamp: float
    relevance: float  # How relevant to current goal
    trace: Optional[ReasoningTrace] = None


class WorkingMemory:
    """
    Short-term memory for recent experiences and reasoning traces
    
    Uses PMFlow's gravitational field to determine relevance and similarity
    """
    
    def __init__(self, capacity: int = 10, decay_rate: float = 0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.items: List[MemoryItem] = []
    
    def add(self, content: torch.Tensor, trace: Optional[ReasoningTrace] = None):
        """Add item to working memory"""
        item = MemoryItem(
            content=content,
            timestamp=time.time(),
            relevance=1.0,
            trace=trace
        )
        
        self.items.append(item)
        
        # Evict oldest if over capacity
        if len(self.items) > self.capacity:
            self.items.pop(0)
    
    def decay(self):
        """Decay relevance of old items"""
        for item in self.items:
            item.relevance *= (1.0 - self.decay_rate)
    
    def recall(self, query: torch.Tensor, top_k: int = 3) -> List[MemoryItem]:
        """Retrieve most relevant items"""
        if not self.items:
            return []
        
        # Flatten query to 1D if needed
        if query.dim() > 1:
            query = query.flatten()
        
        # Compute similarity to query
        similarities = []
        for item in self.items:
            content = item.content.flatten()
            # Ensure same size
            if content.shape[0] != query.shape[0]:
                continue  # Skip incompatible items
            sim = torch.nn.functional.cosine_similarity(
                query.unsqueeze(0), content.unsqueeze(0), dim=1
            ).item()
            similarities.append(sim)
        
        if not similarities:
            return []
        
        similarities = torch.tensor(similarities)
        
        # Weight by relevance
        scores = similarities * torch.tensor([item.relevance for item in self.items if item.content.flatten().shape[0] == query.shape[0]])
        
        # Get top-k
        top_k = min(top_k, len(scores))
        _, indices = torch.topk(scores, top_k)
        
        # Filter to compatible items
        compatible_items = [item for item in self.items if item.content.flatten().shape[0] == query.shape[0]]
        
        return [compatible_items[i] for i in indices.tolist()]
    
    def clear(self):
        """Clear all memory"""
        self.items.clear()


class MetacognitiveMonitor:
    """
    Monitor reasoning process and provide feedback
    
    Watches trajectories to detect:
    - Getting stuck (circular paths)
    - High cognitive effort (long paths)
    - Convergence (reaching goal)
    - Confusion (high variance)
    """
    
    def __init__(self, effort_threshold: float = 5.0, stuck_threshold: int = 3):
        self.effort_threshold = effort_threshold
        self.stuck_threshold = stuck_threshold
        self.history: List[ReasoningTrace] = []
    
    def analyze_trajectory(self, trajectory: torch.Tensor, metrics: Dict) -> Dict[str, Any]:
        """Analyze a reasoning trajectory for issues"""
        analysis = {
            'high_effort': metrics['path_length'] > self.effort_threshold,
            'inefficient': metrics.get('efficiency', 1.0) < 0.5,
            'stuck': self._detect_stuck(trajectory),
            'converged': metrics.get('converged', False),
            'suggestions': []
        }
        
        # Generate suggestions
        if analysis['high_effort']:
            analysis['suggestions'].append("Consider injecting goal intent to reduce search space")
        
        if analysis['stuck']:
            analysis['suggestions'].append("Trajectory is circular - may need to mark current region as hazard")
        
        if analysis['inefficient']:
            analysis['suggestions'].append("Path is meandering - consider stronger attractors")
        
        return analysis
    
    def _detect_stuck(self, trajectory: torch.Tensor) -> bool:
        """Detect if trajectory is going in circles"""
        if len(trajectory) < 5:
            return False
        
        # Check if recent positions are close to earlier positions
        recent = trajectory[-3:]
        earlier = trajectory[:-3]
        
        distances = torch.cdist(recent, earlier)
        min_distances = distances.min(dim=1)[0]
        
        # Stuck if repeatedly returning to same region
        return (min_distances < 0.5).sum() >= 2
    
    def record(self, trace: ReasoningTrace):
        """Record a reasoning trace for learning"""
        self.history.append(trace)


class CognitiveAgent:
    """
    High-level cognitive agent using PMFlow primitives
    
    Composes:
    - Perception (encoding)
    - Working memory (recent context)
    - Reasoning (multi-step thinking)
    - Metacognition (monitoring and adaptation)
    """
    
    def __init__(
        self,
        encoder: PMFlowEmbeddingEncoder,
        memory_capacity: int = 10,
        enable_metacognition: bool = True
    ):
        self.encoder = encoder
        self.memory = WorkingMemory(capacity=memory_capacity)
        self.metacognition = MetacognitiveMonitor() if enable_metacognition else None
        
        # Current goals and constraints
        self.active_goals: List[Tuple[List[str], float]] = []  # (goal, strength)
        self.marked_hazards: List[torch.Tensor] = []
        
    def perceive(self, tokens: List[str]) -> torch.Tensor:
        """
        Encode input into semantic space
        
        Uses working memory to provide context
        """
        # Basic encoding
        embedding = self.encoder.encode(tokens)
        
        # TODO: Could enhance with memory context
        
        return embedding
    
    def think(
        self,
        query: List[str],
        mode: ReasoningMode = ReasoningMode.DIRECT,
        max_iterations: int = 5
    ) -> Tuple[torch.Tensor, ReasoningTrace]:
        """
        Reason about a query
        
        Returns final embedding and trace of reasoning process
        """
        start_time = time.time()
        
        if mode == ReasoningMode.DIRECT:
            result, trace = self._direct_reasoning(query)
        
        elif mode == ReasoningMode.ITERATIVE:
            result, trace = self._iterative_reasoning(query, max_iterations)
        
        elif mode == ReasoningMode.GOAL_DIRECTED:
            result, trace = self._goal_directed_reasoning(query)
        
        elif mode == ReasoningMode.EXPLORATORY:
            result, trace = self._exploratory_reasoning(query, max_iterations)
        
        else:
            raise ValueError(f"Unknown reasoning mode: {mode}")
        
        trace.duration = time.time() - start_time
        
        # Metacognitive analysis
        if self.metacognition:
            analysis = self.metacognition.analyze_trajectory(trace.trajectory, trace.metrics)
            trace.metadata['metacognition'] = analysis
            self.metacognition.record(trace)
        
        # Store in memory
        self.memory.add(result, trace)
        
        return result, trace
    
    def _direct_reasoning(self, query: List[str]) -> Tuple[torch.Tensor, ReasoningTrace]:
        """Single-step encoding"""
        embedding = self.encoder.encode(query)
        
        trace = ReasoningTrace(
            query=query,
            trajectory=embedding.unsqueeze(0),
            metrics={'path_length': 0.0, 'efficiency': 1.0},
            attractors_visited=[],
            hazards_avoided=[],
            mode=ReasoningMode.DIRECT,
            duration=0.0,
            success=True
        )
        
        return embedding, trace
    
    def _iterative_reasoning(
        self, 
        query: List[str], 
        max_iterations: int
    ) -> Tuple[torch.Tensor, ReasoningTrace]:
        """
        Multi-step refinement using trajectory tracing
        
        Each iteration refines the representation by evolving through
        the gravitational field
        """
        if not self.encoder.enable_flow:
            raise ValueError("Iterative reasoning requires enable_flow=True")
        
        # Get trajectory
        trajectory, metrics = self.encoder.trace_trajectory(query)
        
        # Use final position as result
        result = trajectory[-1:].squeeze(0)
        
        # Track which centers were visited
        attractors_visited = self._identify_visited_centers(trajectory)
        
        trace = ReasoningTrace(
            query=query,
            trajectory=trajectory,
            metrics=metrics,
            attractors_visited=attractors_visited,
            hazards_avoided=[],
            mode=ReasoningMode.ITERATIVE,
            duration=0.0,
            success=metrics.get('converged', True)
        )
        
        return result, trace
    
    def _goal_directed_reasoning(
        self, 
        query: List[str]
    ) -> Tuple[torch.Tensor, ReasoningTrace]:
        """
        Goal-directed reasoning with intent injection
        
        Uses active goals to bias the semantic flow
        """
        if not self.encoder.enable_flow:
            raise ValueError("Goal-directed reasoning requires enable_flow=True")
        
        # Inject all active goals
        for goal_tokens, strength in self.active_goals:
            self.encoder.inject_intent(goal_tokens, strength=strength)
        
        # Reason with goals active
        trajectory, metrics = self.encoder.trace_trajectory(query)
        result = trajectory[-1:].squeeze(0)
        
        # Clear goals (unless persistent)
        self.encoder.clear_intent()
        
        attractors_visited = self._identify_visited_centers(trajectory)
        
        trace = ReasoningTrace(
            query=query,
            trajectory=trajectory,
            metrics=metrics,
            attractors_visited=attractors_visited,
            hazards_avoided=[],
            mode=ReasoningMode.GOAL_DIRECTED,
            duration=0.0,
            success=True
        )
        
        return result, trace
    
    def _exploratory_reasoning(
        self, 
        query: List[str],
        max_iterations: int
    ) -> Tuple[torch.Tensor, ReasoningTrace]:
        """
        Exploratory reasoning - try multiple paths
        
        Useful when uncertain about best approach
        """
        # Try reasoning with small random perturbations
        trajectories = []
        
        for i in range(min(3, max_iterations)):
            # Add small exploration noise to field
            # (would need to implement perturbation in encoder)
            trajectory, metrics = self.encoder.trace_trajectory(query)
            trajectories.append((trajectory, metrics))
        
        # Pick trajectory with best efficiency
        best_traj, best_metrics = max(trajectories, key=lambda x: x[1].get('efficiency', 0))
        result = best_traj[-1:].squeeze(0)
        
        trace = ReasoningTrace(
            query=query,
            trajectory=best_traj,
            metrics=best_metrics,
            attractors_visited=self._identify_visited_centers(best_traj),
            hazards_avoided=[],
            mode=ReasoningMode.EXPLORATORY,
            duration=0.0,
            success=True
        )
        
        return result, trace
    
    def _identify_visited_centers(self, trajectory: torch.Tensor) -> List[int]:
        """Identify which gravitational centers the trajectory passed near"""
        if not hasattr(self.encoder, 'field'):
            return []
        
        # Get centers from the field
        if hasattr(self.encoder.field, 'fine_field'):
            centers = self.encoder.field.fine_field.centers
        else:
            centers = self.encoder.field.centers
        
        visited = []
        for point in trajectory:
            distances = torch.norm(centers - point, dim=1)
            closest = distances.argmin().item()
            if closest not in visited:
                visited.append(closest)
        
        return visited
    
    def set_goal(self, goal_tokens: List[str], strength: float = 0.5, persistent: bool = False):
        """
        Set a reasoning goal
        
        Future reasoning will be biased toward this goal concept
        """
        self.active_goals.append((goal_tokens, strength))
    
    def clear_goals(self):
        """Clear all active goals"""
        self.active_goals.clear()
        self.encoder.clear_intent()
    
    def mark_hazard(self, location: torch.Tensor, radius: float = 1.0):
        """Mark a region of semantic space to avoid"""
        self.marked_hazards.append(location)
        # TODO: Apply to encoder's field if accessible
    
    def explain_reasoning(self, trace: ReasoningTrace) -> str:
        """
        Generate human-readable explanation of reasoning process
        """
        explanation = f"Reasoning Mode: {trace.mode.value}\n"
        explanation += f"Query: {' '.join(trace.query)}\n"
        explanation += f"Path Length (Mental Effort): {trace.metrics['path_length']:.3f}\n"
        explanation += f"Efficiency: {trace.metrics.get('efficiency', 0):.3f}\n"
        
        if trace.attractors_visited:
            explanation += f"Semantic Concepts Visited: {trace.attractors_visited}\n"
        
        if 'metacognition' in trace.metadata:
            meta = trace.metadata['metacognition']
            if meta['suggestions']:
                explanation += "\nMetacognitive Insights:\n"
                for suggestion in meta['suggestions']:
                    explanation += f"  - {suggestion}\n"
        
        return explanation


# Example usage demonstration
def demo_cognitive_agent():
    """Demonstrate the cognitive architecture"""
    
    # Create agent
    encoder = PMFlowEmbeddingEncoder(
        dimension=96,
        latent_dim=48,
        enable_flow=True
    )
    
    agent = CognitiveAgent(encoder, memory_capacity=10)
    
    # Example 1: Direct reasoning
    print("=== Example 1: Direct Reasoning ===")
    result, trace = agent.think(["machine", "learning"], mode=ReasoningMode.DIRECT)
    print(agent.explain_reasoning(trace))
    
    # Example 2: Goal-directed reasoning
    print("\n=== Example 2: Goal-Directed Reasoning ===")
    agent.set_goal(["neural", "networks"], strength=0.7)
    result, trace = agent.think(["artificial", "intelligence"], mode=ReasoningMode.GOAL_DIRECTED)
    print(agent.explain_reasoning(trace))
    agent.clear_goals()
    
    # Example 3: Iterative reasoning
    print("\n=== Example 3: Iterative Reasoning ===")
    result, trace = agent.think(["deep", "learning"], mode=ReasoningMode.ITERATIVE)
    print(agent.explain_reasoning(trace))
    
    # Check metacognition
    if trace.metadata.get('metacognition'):
        meta = trace.metadata['metacognition']
        print(f"High effort: {meta['high_effort']}")
        print(f"Stuck: {meta['stuck']}")


if __name__ == "__main__":
    demo_cognitive_agent()
