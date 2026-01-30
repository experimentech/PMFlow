import torch
import pytest
from pmflow.core.pmflow import ParallelPMField

def test_trajectory_shape():
    """Verify that return_trajectory returns the full history with correct shape."""
    batch_size = 4
    d_latent = 8
    steps = 5
    
    pm = ParallelPMField(d_latent=d_latent, steps=steps)
    z = torch.randn(batch_size, d_latent)
    
    # Test without trajectory (default)
    out = pm(z)
    assert out.shape == (batch_size, d_latent)
    
    # Test with trajectory
    out_traj = pm(z, return_trajectory=True)
    # Shape should be (Batch, Steps + 1, Latent)
    assert out_traj.shape == (batch_size, steps + 1, d_latent)
    
    # Verify the first point is the input
    assert torch.allclose(out_traj[:, 0, :], z)
    
    # Verify the last point matches the non-trajectory output
    # Note: re-running pm(z) might produce slightly different results if there were randomness 
    # (e.g. dropout, though not present here) or statefulness. 
    # ParallelPMField is deterministic for fixed parameters.
    assert torch.allclose(out_traj[:, -1, :], out)

def test_flow_impact():
    """Verify that enabling flow changes the trajectory."""
    d_latent = 8
    # Initialize with flow enabled
    pm = ParallelPMField(d_latent=d_latent, enable_flow=True)
    
    # Set omegas to something non-zero to ensure flow happens
    with torch.no_grad():
        pm.omegas.fill_(1.0)
        # Ensure centers and input aren't zero so r is non-zero
        pm.centers.normal_()
        
    z = torch.randn(2, d_latent)
    
    # Run with flow enabled
    out_with_flow = pm(z)
    
    # Run with flow disabled (temporarily disable flag)
    pm.enable_flow = False
    out_no_flow = pm(z)
    
    # Outputs should differ if flow is working
    # (Assuming beta isn't dominating everything and dt is sufficient)
    diff = torch.norm(out_with_flow - out_no_flow)
    assert diff > 1e-5, "Flow field did not affect the output tensor"

def test_swirl_direction():
    """
    Verify the specific rotational logic: (x, y) -> (-y, x).
    We use d_latent=2 for clarity.
    """
    # Beta=0 removes gravity, isolating flow
    pm = ParallelPMField(d_latent=2, n_centers=1, enable_flow=True, beta=0.0) 
    
    with torch.no_grad():
        # Center at origin
        pm.centers.fill_(0.0)
        # Omega = Positive
        pm.omegas.fill_(10.0) 
        pm.dt = 0.1
    
    # Particle at (1, 0)
    z = torch.tensor([[1.0, 0.0]])
    
    # Logic in code:
    # rvec = z - center = (1, 0)
    # rvec_rotated = (-y, x) = (0, 1)
    # flow ~ omega * rvec_rotated * (1/r)
    # flow direction should be +y, i.e., (0, positive)
    
    flow = pm.vectorized_flow_field(z)
    
    print(f"Flow vector: {flow}")
    
    # x component should be 0 (or very close)
    assert torch.abs(flow[0, 0]) < 1e-5
    # y component should be positive
    assert flow[0, 1] > 0.0

def test_odd_dimension_stability():
    """Verify flow calculation works for odd dimensions (latent space edge case)."""
    d_latent = 3
    pm = ParallelPMField(d_latent=d_latent, enable_flow=True)
    z = torch.randn(2, d_latent)
    
    # Should not crash
    out = pm(z)
    assert out.shape == (2, d_latent)
    
    # Check flow vector specifically
    flow = pm.vectorized_flow_field(z)
    assert flow.shape == (2, d_latent)
    # The last dimension in 3D (index 2) should be 0 because code says:
    # if D % 2 == 1: rvec_rotated[:, :, -1] = 0
    assert torch.all(flow[:, -1] == 0)

if __name__ == "__main__":
    # helper to run without pytest
    try:
        test_trajectory_shape()
        test_flow_impact()
        test_swirl_direction()
        test_odd_dimension_stability()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        raise
