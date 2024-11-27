import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

class DiffusionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.norm1(x)))
        h = h + t_emb.unsqueeze(-1)
        h = self.conv2(F.relu(self.norm2(h)))
        return h + x

class ScoreBasedDiffusion(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02
    ):
        super().__init__()
        
        # Diffusion parameters
        self.num_timesteps = num_timesteps
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, num_timesteps))
        alphas = 1 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Sequence processing
        self.input_conv = nn.Conv1d(5, hidden_size, 1)  # 5 for ACGTN
        
        # Diffusion blocks
        self.blocks = nn.ModuleList([
            DiffusionBlock(hidden_size)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_conv = nn.Conv1d(hidden_size, 5, 1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Initial processing
        h = self.input_conv(x.transpose(1, 2))
        
        # Process through blocks
        for block in self.blocks:
            h = block(h, t_emb)
            
        # Output
        return self.output_conv(h).transpose(1, 2)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample from q(x_t | x_0)."""
        alphas_cumprod_t = self.alphas_cumprod[t]
        return torch.sqrt(alphas_cumprod_t)[:, None, None] * x_start + \
               torch.sqrt(1 - alphas_cumprod_t)[:, None, None] * noise
               
    def p_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample from p(x_{t-1} | x_t)."""
        with torch.no_grad():
            score = self(x, t)
            alpha_t = self.alphas_cumprod[t]
            sigma_t = torch.sqrt(1 - alpha_t)
            z = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
            x_prev = (1 / torch.sqrt(alpha_t))[:, None, None] * \
                     (x - (sigma_t)[:, None, None] * score) + \
                     torch.sqrt(self.betas[t])[:, None, None] * z
            return x_prev
            
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """Generate samples."""
        self.eval()
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch)
            
        return x
    
    def training_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Perform a training step."""
        x_start = batch
        t = torch.randint(0, self.num_timesteps, (x_start.size(0),), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self(x_noisy, t)
        
        loss = F.mse_loss(predicted_noise, noise)
        return predicted_noise, loss.item()