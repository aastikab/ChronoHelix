# Import specific classes
from data import DNAPreprocessor
from data import AncientDNADataset
from data import DataModule

# Import constants
from data import NUCLEOTIDES, DEFAULT_K

# Use utility functions
from data import validate_sequence, get_default_params
from .preprocessing import DNAPreprocessor
from .data_loader import AncientDNADataset, DataModule

__all__ = [
    'DNAPreprocessor',
    'AncientDNADataset',
    'DataModule'
]

# Version info
__version__ = '0.1.0'

# Module level constants
NUCLEOTIDES = ['A', 'C', 'G', 'T', 'N']
DEFAULT_K = 3  # Default k-mer size
DEFAULT_MIN_FRAGMENT_LENGTH = 10
DEFAULT_MAX_FRAGMENT_LENGTH = 50
DEFAULT_ERROR_RATE = 0.1

# Module level utility functions
def get_default_params():
    """Return default parameters for data processing."""
    return {
        'k_mer_size': DEFAULT_K,
        'min_fragment_length': DEFAULT_MIN_FRAGMENT_LENGTH,
        'max_fragment_length': DEFAULT_MAX_FRAGMENT_LENGTH,
        'error_rate': DEFAULT_ERROR_RATE
    }

def validate_sequence(sequence: str) -> bool:
    """
    Validate if a sequence contains only valid nucleotides.
    
    Args:
        sequence (str): DNA sequence to validate
        
    Returns:
        bool: True if sequence is valid, False otherwise
    """
    return all(nuc in NUCLEOTIDES for nuc in sequence.upper())