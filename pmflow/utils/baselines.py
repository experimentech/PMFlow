import torch
import torch.nn as nn
from .pmflow import ParallelPMField

class CNNBaseline(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*7*7, 256), nn.ReLU(),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.net(x)

class GRUBaseline(nn.Module):
    def __init__(self, hidden=128, n_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size=28, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        seq = x.squeeze(1)  # (B, 28, 28)
        out, _ = self.gru(seq)
        return self.fc(out[:, -1, :])

class MLPBaseline(nn.Module):
    """Enhanced MLP baseline for v0.2.0 comparisons."""
    def __init__(self, hidden_sizes=[256, 128], n_classes=10):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(28*28, hidden_sizes[0]), nn.ReLU()]
        
        for i in range(len(hidden_sizes) - 1):
            layers.extend([
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
        
        layers.append(nn.Linear(hidden_sizes[-1], n_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class PMFlowCNN(nn.Module):
    """CNN with PMFlow latent field integration (v0.2.0).

    Pipeline: Conv -> Flatten -> Linear to latent -> ParallelPMField -> Linear head
    This model applies PMFlow physics to a compact latent before classification.
    """
    def __init__(self,
                 n_classes: int = 10,
                 d_latent: int = 64,
                 n_centers: int = 32,
                 pm_steps: int = 4,
                 dt: float = 0.15,
                 beta: float = 1.2,
                 clamp: float = 3.0,
                 temporal_parallel: bool = True,
                 chunk_size: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )
        self.to_latent = nn.Linear(128 * 4 * 4, d_latent)
        self.pm = ParallelPMField(
            d_latent=d_latent,
            n_centers=n_centers,
            steps=pm_steps,
            dt=dt,
            beta=beta,
            clamp=clamp,
            temporal_parallel=temporal_parallel,
            chunk_size=chunk_size,
        )
        self.head = nn.Linear(d_latent, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        z = self.conv(x)
        z = self.to_latent(z)
        z = self.pm(z)
        return self.head(z)