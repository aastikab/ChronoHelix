import torch
import numpy as np
from typing import List, Dict, Tuple
from Bio import pairwise2
from collections import defaultdict

class SequenceEvaluator:
    def __init__(self):
        self.nucleotides = ['A', 'C', 'G', 'T', 'N']
        self.nuc_to_idx = {nuc: idx for idx, nuc in enumerate(self.nucleotides)}
        
    def _onehot_to_sequence(self, onehot: torch.Tensor) -> str:
        """Convert one-hot encoded tensor to sequence string."""
        indices = torch.argmax(onehot, dim=-1)
        return ''.join(self.nucleotides[idx] for idx in indices)
    
    def calculate_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Calculate various accuracy metrics."""
        pred_seq = self._onehot_to_sequence(pred)
        target_seq = self._onehot_to_sequence(target)
        
        # Nucleotide-level accuracy
        correct_nucs = sum(p == t for p, t in zip(pred_seq, target_seq))
        nuc_accuracy = correct_nucs / len(target_seq)
        
        # Position-specific accuracy
        pos_accuracy = defaultdict(float)
        for pos, (p, t) in enumerate(zip(pred_seq, target_seq)):
            pos_accuracy[pos] += int(p == t)
            
        return {
            'nucleotide_accuracy': nuc_accuracy,
            'position_accuracy': dict(pos_accuracy)
        }
    
    def calculate_edit_distance(self, pred: str, target: str) -> int:
        """Calculate Levenshtein distance between sequences."""
        if len(pred) == 0: return len(target)
        if len(target) == 0: return len(pred)
        
        matrix = [[0 for _ in range(len(target) + 1)] for _ in range(len(pred) + 1)]
        
        for i in range(len(pred) + 1):
            matrix[i][0] = i
        for j in range(len(target) + 1):
            matrix[0][j] = j
            
        for i in range(1, len(pred) + 1):
            for j in range(1, len(target) + 1):
                if pred[i-1] == target[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,    # deletion
                        matrix[i][j-1] + 1,    # insertion
                        matrix[i-1][j-1] + 1   # substitution
                    )
                    
        return matrix[len(pred)][len(target)]
    
    def calculate_alignment_score(self, pred: str, target: str) -> float:
        """Calculate sequence alignment score."""
        alignments = pairwise2.align.globalxx(pred, target)
        if alignments:
            return alignments[0].score / max(len(pred), len(target))
        return 0.0
    
    def evaluate_batch(self, 
                      predictions: torch.Tensor, 
                      targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate a batch of predictions."""
        batch_metrics = defaultdict(list)
        
        for pred, target in zip(predictions, targets):
            pred_seq = self._onehot_to_sequence(pred)
            target_seq = self._onehot_to_sequence(target)
            
            # Calculate all metrics
            accuracy_metrics = self.calculate_accuracy(pred, target)
            edit_dist = self.calculate_edit_distance(pred_seq, target_seq)
            align_score = self.calculate_alignment_score(pred_seq, target_seq)
            
            # Store metrics
            batch_metrics['nucleotide_accuracy'].append(accuracy_metrics['nucleotide_accuracy'])
            batch_metrics['edit_distance'].append(edit_dist)
            batch_metrics['alignment_score'].append(align_score)
        
        # Calculate mean metrics
        return {
            'mean_nucleotide_accuracy': np.mean(batch_metrics['nucleotide_accuracy']),
            'mean_edit_distance': np.mean(batch_metrics['edit_distance']),
            'mean_alignment_score': np.mean(batch_metrics['alignment_score'])
        }
    
    def analyze_errors(self, pred: str, target: str) -> Dict[str, Dict[str, int]]:
        """Analyze types of errors in prediction."""
        error_types = {
            'substitutions': defaultdict(int),
            'insertions': 0,
            'deletions': 0
        }
        
        # Align sequences
        alignment = pairwise2.align.globalxx(pred, target)[0]
        aligned_pred, aligned_target = alignment.seqA, alignment.seqB
        
        # Analyze errors
        for p, t in zip(aligned_pred, aligned_target):
            if p == '-':
                error_types['deletions'] += 1
            elif t == '-':
                error_types['insertions'] += 1
            elif p != t:
                error_types['substitutions'][f"{t}->{p}"] += 1
                
        return error_types