from .visualization import DNAVisualizer
from .evaluation import SequenceEvaluator

__all__ = [
    'DNAVisualizer',
    'SequenceEvaluator'
]

# Version info
__version__ = '0.1.0'

# Module level constants
METRICS = [
    'sequence_accuracy',
    'nucleotide_accuracy',
    'edit_distance',
    'coverage'
]

VISUALIZATION_TYPES = [
    'quality_distribution',
    'kmer_graph',
    'training_progress',
    'sequence_comparison'
]

# Utility functions
def get_available_metrics():
    """Returns list of available evaluation metrics."""
    return METRICS.copy()

def get_visualization_types():
    """Returns list of available visualization types."""
    return VISUALIZATION_TYPES.copy()

def calculate_gc_content(sequence: str) -> float:
    """
    Calculate GC content of a DNA sequence.
    
    Args:
        sequence (str): DNA sequence
        
    Returns:
        float: GC content percentage
    """
    if not sequence:
        return 0.0
    gc_count = sequence.upper().count('G') + sequence.upper().count('C')
    return (gc_count / len(sequence)) * 100

def get_sequence_stats(sequence: str) -> dict:
    """
    Get basic statistics about a DNA sequence.
    
    Args:
        sequence (str): DNA sequence
        
    Returns:
        dict: Dictionary containing sequence statistics
    """
    return {
        'length': len(sequence),
        'gc_content': calculate_gc_content(sequence),
        'n_count': sequence.upper().count('N'),
        'unique_kmers': len(set(sequence[i:i+3] for i in range(len(sequence)-2)))
    }