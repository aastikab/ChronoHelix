import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import torch
from torch_geometric.utils import to_networkx
import numpy as np
from typing import List, Optional

class DNAVisualizer:
    def __init__(self):
        self.nucleotides = ['A', 'C', 'G', 'T', 'N']
        
    def plot_sequence_quality(self, sequence: str, title: Optional[str] = None):
        """Plot sequence quality distribution."""
        nucleotide_counts = {n: sequence.count(n) for n in self.nucleotides}
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(nucleotide_counts.keys()), 
                   y=list(nucleotide_counts.values()))
        plt.title(title or "Nucleotide Distribution")
        plt.xlabel("Nucleotide")
        plt.ylabel("Count")
        plt.show()
        
    def visualize_kmer_graph(self, data, k: int = 3):
        """Visualize k-mer graph structure."""
        G = to_networkx(data, to_undirected=True)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=500, node_color='lightblue',
                with_labels=True, font_size=8)
        plt.title(f"{k}-mer Graph Visualization")
        plt.show()
        
    def plot_training_progress(self, losses: List[float], 
                             val_losses: Optional[List[float]] = None):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.title("Training Progress")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def compare_sequences(self, original: str, reconstructed: str, 
                         window_size: int = 50):
        """Compare original and reconstructed sequences."""
        if len(original) != len(reconstructed):
            raise ValueError("Sequences must be of equal length")
            
        # Calculate similarity score
        matches = sum(1 for a, b in zip(original, reconstructed) if a == b)
        similarity = matches / len(original) * 100
        
        # Visualize differences
        differences = [i for i, (a, b) in enumerate(zip(original, reconstructed)) 
                      if a != b]
        
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(original)), 
                [1 if i in differences else 0 for i in range(len(original))],
                'r.', markersize=1)
        plt.title(f"Sequence Differences (Similarity: {similarity:.2f}%)")
        plt.xlabel("Position")
        plt.ylabel("Difference")
        plt.show()