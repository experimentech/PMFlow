import torch
import sys
import math
from pathlib import Path

# Add parent directory to path to find pmflow
sys.path.append(str(Path(__file__).parent.parent))

from pmflow.core.pmflow import ParallelPMField

def render_grid(centers, trajectory, title="Simulation", size=20):
    """Render a 2D ascii grid of the field and trajectory."""
    grid = [[' ' for _ in range(size)] for _ in range(size)]
    
    def to_grid(x, y):
        # Map range [-1.5, 1.5] to [0, size-1]
        gx = int((x + 1.5) / 3.0 * (size - 1))
        gy = int((y + 1.5) / 3.0 * (size - 1))
        # Flip Y for rendering
        return gx, size - 1 - gy

    # Plot Centers
    for i in range(centers.shape[0]):
        cx, cy = centers[i].tolist()
        gx, gy = to_grid(cx, cy)
        if 0 <= gx < size and 0 <= gy < size:
            if i == 0:
                grid[gy][gx] = 'G' # Goal
            else:
                grid[gy][gx] = 'O' # Obstacle

    # Plot Trajectory
    prev_gx, prev_gy = -1, -1
    for t, point in enumerate(trajectory):
        px, py = point.tolist()
        gx, gy = to_grid(px, py)
        
        if 0 <= gx < size and 0 <= gy < size:
            char = '.'
            if t == 0: char = 'S' # Start
            elif t == len(trajectory)-1: char = 'E' # End
            
            if grid[gy][gx] == ' ':
                grid[gy][gx] = char

    print(f"\n=== {title} ===")
    print("Legend: S=Start, E=End, G=Goal (Top Right), O=Obstacle (Middle)")
    print("+" + "-"*size + "+")
    for row in grid:
        print("|" + "".join(row) + "|")
    print("+" + "-"*size + "+")

def demo_physics():
    print("Initializing Agentic Physics Demo...")
    
    # 2D field for visualization
    d_latent = 2
    steps = 15 # Long thought process
    dt = 0.1
    
    # Enable Flow!
    pm = ParallelPMField(d_latent=d_latent, n_centers=2, steps=steps, dt=dt, enable_flow=True)
    
    # Setup the Scenario
    with torch.no_grad():
        # Center 0: GOAL (Top Right)
        pm.centers[0] = torch.tensor([0.8, 0.8])
        pm.mus[0] = 1.0     # Metric Gravity (Standard attraction)
        
        # Center 1: OBSTACLE (Middle, blocking the path)
        pm.centers[1] = torch.tensor([0.0, 0.0]) 
        pm.mus[1] = 1.2     # Stronger Gravity (Distracting thought)
        
        # Start Condition (Bottom Left)
        start_pos = torch.tensor([[-0.9, -0.9]])
    
    # Simulation 1: PASSIVE (No Willpower/Flow)
    print("\n[Case 1] Passive Retrieval (Gravity Only)")
    with torch.no_grad():
        pm.enable_flow = False
        traj_passive = pm(start_pos, return_trajectory=True)[0] # Get single batch item
    
    render_grid(pm.centers, traj_passive, title="Passive: Getting Stuck")
    
    # Simulation 2: AGENTIC (With Intent/Flow)
    print("\n[Case 2] Agentic Reasoning (Gravity + Flow Field)")
    with torch.no_grad():
        pm.steps = 40  # Meaningful thought takes time
        pm.enable_flow = True
        
        # AGENTIC STRATEGY: "The Gravitational Slingshot"
        # 1. Spin the Obstacle gently to create a 'passable' orbit (avoid collision)
        # 2. Increase Goal gravity to ensure capture once past the event horizon
        
        pm.omegas.fill_(0.0) 
        pm.omegas[1] = -2.5  # Gentle clockwise swirl
        
        pm.mus[0] = 2.0      # Stronger desire for the goal
        
        traj_active = pm(start_pos, return_trajectory=True)[0]
        
    render_grid(pm.centers, traj_active, title="Agentic: Curving Around")
    
    # Analyze "Mental Effort" (Action)
    diffs = traj_active[1:] - traj_active[:-1]
    action = torch.sum(torch.norm(diffs, dim=1)).item()
    
    # Check if we EVER reached the goal (Aha! moment)
    # In a cognitive system, we stop thinking when we find the answer.
    dists = torch.norm(traj_active - pm.centers[0], dim=1)
    min_dist = torch.min(dists).item()
    reached_goal = min_dist < 0.5
    
    print(f"\nCognitive Metrics:")
    print(f"Path Length (Mental Effort): {action:.4f}")
    print(f"Closest Approach: {min_dist:.4f}")
    print(f"Did we reach goal? {'Yes' if reached_goal else 'No'}")

if __name__ == "__main__":
    demo_physics()
