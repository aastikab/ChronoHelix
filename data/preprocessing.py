import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
import torch
from typing import List, Tuple, Dict

class DNAPreprocessor:
    def __init__(self):
        self.nucleotides = ['A', 'C', 'G', 'T', 'N']  # N for unknown
        self.nuc_to_idx = {nuc: idx for idx, nuc in enumerate(self.nucleotides)}
        
    def sequence_to_one_hot(self, sequence: str) -> np.ndarray:
        """Convert DNA sequence to one-hot encoding."""
        sequence = sequence.upper()
        one_hot = np.zeros((len(sequence), len(self.nucleotides)))
        
        for i, nuc in enumerate(sequence):
            if nuc in self.nuc_to_idx:
                one_hot[i, self.nuc_to_idx[nuc]] = 1
            else:
                one_hot[i, self.nuc_to_idx['N']] = 1
                
        return one_hot
    
    def create_fragments(self, sequence: str, min_length: int = 10, 
                        max_length: int = 50) -> List[str]:
        """Simulate DNA fragmentation."""
        fragments = []
        seq_length = len(sequence)
        
        current_pos = 0
        while current_pos < seq_length:
            # Random fragment length
            frag_length = np.random.randint(min_length, max_length)
            if current_pos + frag_length > seq_length:
                frag_length = seq_length - current_pos
                
            fragment = sequence[current_pos:current_pos + frag_length]
            fragments.append(fragment)
            current_pos += frag_length
            
        return fragments
    
    def add_noise(self, sequence: str, error_rate: float = 0.1) -> str:
        """Simulate DNA degradation by adding noise."""
        sequence = list(sequence.upper())
        for i in range(len(sequence)):
            if np.random.random() < error_rate:
                if np.random.random() < 0.5:  # 50% chance of mutation
                    valid_nucs = [n for n in self.nucleotides if n != sequence[i] and n != 'N']
                    sequence[i] = np.random.choice(valid_nucs)
                else:  # 50% chance of unknown
                    sequence[i] = 'N'
                    
        return ''.join(sequence)
    
    def create_kmer_graph(self, sequence: str, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create a k-mer graph from DNA sequence."""
        # Generate k-mers
        kmers = [sequence[i:i+k] for i in range(len(sequence)-k+1)]
        unique_kmers = list(set(kmers))
        kmer_to_idx = {kmer: idx for idx, kmer in enumerate(unique_kmers)}
        
        # Create edges between overlapping k-mers
        edges = []
        for i in range(len(kmers)-1):
            src_idx = kmer_to_idx[kmers[i]]
            dst_idx = kmer_to_idx[kmers[i+1]]
            edges.append([src_idx, dst_idx])
            
        # Create node features (one-hot encoding of first nucleotide)
        node_features = torch.zeros((len(unique_kmers), len(self.nucleotides)))
        for i, kmer in enumerate(unique_kmers):
            node_features[i] = torch.tensor(self.sequence_to_one_hot(kmer[0]))
            
        edge_index = torch.tensor(edges).t().contiguous()
        
        return node_features, edge_index