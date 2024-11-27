import torch
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class DataConfig:
    fasta_files: List[str]
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    min_fragment_length: int = 10
    max_fragment_length: int = 50
    k_mer_size: int = 3
    error_rate: float = 0.1

@dataclass
class GNNConfig:
    in_channels: int = 5  # ACGTN
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 0.01

@dataclass
class DiffusionConfig:
    sequence_length: int = 100
    hidden_size: int = 128
    num_layers: int = 4
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    learning_rate: float = 0.0001
    weight_decay: float = 0.01

@dataclass
class TrainingConfig:
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs: int = 100
    save_interval: int = 10
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    model_name: str = 'chronohelix'
    early_stopping_patience: int = 10
    grad_clip: float = 1.0
    
class ChronoHelixConfig:
    def __init__(
        self,
        data_config: DataConfig,
        gnn_config: GNNConfig,
        diffusion_config: DiffusionConfig,
        training_config: TrainingConfig
    ):
        self.data = data_config
        self.gnn = gnn_config
        self.diffusion = diffusion_config
        self.training = training_config
        
    @classmethod
    def from_dict(cls, config_dict: dict):
        """Create config from dictionary."""
        return cls(
            DataConfig(**config_dict.get('data', {})),
            GNNConfig(**config_dict.get('gnn', {})),
            DiffusionConfig(**config_dict.get('diffusion', {})),
            TrainingConfig(**config_dict.get('training', {}))
        )
    
    def save(self, path: str):
        """Save config to file."""
        import json
        from dataclasses import asdict
        
        config_dict = {
            'data': asdict(self.data),
            'gnn': asdict(self.gnn),
            'diffusion': asdict(self.diffusion),
            'training': {k: str(v) if isinstance(v, torch.device) else v 
                        for k, v in asdict(self.training).items()}
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    @classmethod
    def load(cls, path: str):
        """Load config from file."""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
            
        # Convert device string back to torch.device
        if 'training' in config_dict:
            if 'device' in config_dict['training']:
                config_dict['training']['device'] = torch.device(
                    config_dict['training']['device']
                )
                
        return cls.from_dict(config_dict)