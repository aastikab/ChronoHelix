from .gnn import DNAGraphNetwork
from .diffusion import ScoreBasedDiffusion

__all__ = [
    'DNAGraphNetwork',
    'ScoreBasedDiffusion'
]

# Version info
__version__ = '0.1.0'

# Module level constants
GNN_DEFAULTS = {
    'hidden_channels': 64,
    'num_layers': 3,
    'dropout': 0.1,
    'learning_rate': 0.001
}

DIFFUSION_DEFAULTS = {
    'num_timesteps': 1000,
    'beta_start': 0.0001,
    'beta_end': 0.02,
    'hidden_size': 128,
    'num_layers': 4
}

def get_model_config(model_type: str = 'gnn') -> dict:
    """
    Get default configuration for specified model type.
    
    Args:
        model_type (str): Either 'gnn' or 'diffusion'
        
    Returns:
        dict: Default configuration parameters
    """
    if model_type.lower() == 'gnn':
        return GNN_DEFAULTS.copy()
    elif model_type.lower() == 'diffusion':
        return DIFFUSION_DEFAULTS.copy()
    else:
        raise ValueError(f"Unknown model type: {model_type}")