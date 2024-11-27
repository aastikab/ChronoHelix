from Bio import SeqIO
import torch
from torch_geometric.data import Data, Dataset
from typing import List, Optional
import numpy as np
from .preprocessing import DNAPreprocessor

class AncientDNADataset(Dataset):
    def __init__(self, fasta_files: List[str], transform=None):
        super().__init__(transform)
        self.fasta_files = fasta_files
        self.preprocessor = DNAPreprocessor()
        self.sequences = self._load_sequences()
        
    def _load_sequences(self) -> List[str]:
        """Load sequences from FASTA files."""
        sequences = []
        for fasta_file in self.fasta_files:
            for record in SeqIO.parse(fasta_file, "fasta"):
                sequences.append(str(record.seq))
        return sequences
    
    def len(self) -> int:
        return len(self.sequences)
    
    def get(self, idx: int) -> Data:
        """Get a single graph-structured data point."""
        sequence = self.sequences[idx]
        
        # Create fragments and add noise to simulate ancient DNA
        fragments = self.preprocessor.create_fragments(sequence)
        noisy_fragments = [self.preprocessor.add_noise(f) for f in fragments]
        
        # Create graph representation for each fragment
        fragment_graphs = []
        for fragment in noisy_fragments:
            node_features, edge_index = self.preprocessor.create_kmer_graph(fragment)
            fragment_graphs.append(Data(x=node_features, edge_index=edge_index))
            
        # Combine fragment graphs into a single graph
        combined_x = torch.cat([g.x for g in fragment_graphs], dim=0)
        combined_edge_index = torch.cat([g.edge_index + i * g.x.size(0) 
                                       for i, g in enumerate(fragment_graphs)], dim=1)
        
        # Create the target sequence one-hot encoding
        target = torch.tensor(self.preprocessor.sequence_to_one_hot(sequence))
        
        return Data(x=combined_x, 
                   edge_index=combined_edge_index, 
                   y=target)

class DataModule:
    def __init__(self, fasta_files: List[str], batch_size: int = 32,
                 train_split: float = 0.8, val_split: float = 0.1):
        self.fasta_files = fasta_files
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        
    def setup(self):
        """Set up train, validation, and test datasets."""
        dataset = AncientDNADataset(self.fasta_files)
        
        # Split dataset
        n_samples = len(dataset)
        n_train = int(n_samples * self.train_split)
        n_val = int(n_samples * self.val_split)
        n_test = n_samples - n_train - n_val
        
        self.train_dataset, self.val_dataset, self.test_dataset = \
            torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )