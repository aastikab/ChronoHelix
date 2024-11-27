import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Tuple

class DNAGraphNetwork(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # Input layer
        self.input_conv = GCNConv(in_channels, hidden_channels)
        
        # Hidden layers
        self.convs = torch.nn.ModuleList([
            GCNConv(hidden_channels, hidden_channels)
            for _ in range(num_layers - 1)
        ])
        
        # Output layers
        self.sequence_predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 5)  # 5 for ACGTN
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch) -> torch.Tensor:
        # Initial convolution
        x = self.input_conv(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Hidden layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Sequence prediction
        x = self.sequence_predictor(x)
        
        return x
    
    def training_step(self, batch) -> Tuple[torch.Tensor, float]:
        """Perform a training step."""
        pred = self(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(pred, batch.y)
        return pred, loss.item()
    
    @torch.no_grad()
    def predict(self, batch) -> torch.Tensor:
        """Make predictions."""
        self.eval()
        pred = self(batch.x, batch.edge_index, batch.batch)
        return F.softmax(pred, dim=-1)