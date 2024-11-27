import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
import time
from tqdm import tqdm
import json
from typing import Dict, Any, Optional

from models import DNAGraphNetwork, ScoreBasedDiffusion
from data import DataModule
from utils import SequenceEvaluator
from config import ChronoHelixConfig

class Trainer:
    def __init__(self, config: ChronoHelixConfig):
        self.config = config
        self.device = config.training.device
        
        # Setup logging
        self.setup_logging()
        
        # Initialize models
        self.gnn = DNAGraphNetwork(
            in_channels=config.gnn.in_channels,
            hidden_channels=config.gnn.hidden_channels,
            num_layers=config.gnn.num_layers,
            dropout=config.gnn.dropout
        ).to(self.device)
        
        self.diffusion = ScoreBasedDiffusion(
            sequence_length=config.diffusion.sequence_length,
            hidden_size=config.diffusion.hidden_size,
            num_layers=config.diffusion.num_layers,
            num_timesteps=config.diffusion.num_timesteps,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end
        ).to(self.device)
        
        # Initialize optimizers
        self.gnn_optimizer = optim.Adam(
            self.gnn.parameters(),
            lr=config.gnn.learning_rate,
            weight_decay=config.gnn.weight_decay
        )
        
        self.diffusion_optimizer = optim.Adam(
            self.diffusion.parameters(),
            lr=config.diffusion.learning_rate,
            weight_decay=config.diffusion.weight_decay
        )
        
        # Initialize data
        self.data_module = DataModule(
            fasta_files=config.data.fasta_files,
            batch_size=config.data.batch_size,
            train_split=config.data.train_split,
            val_split=config.data.val_split
        )
        self.data_module.setup()
        
        # Initialize evaluator
        self.evaluator = SequenceEvaluator()
        
        # Initialize tensorboard
        self.writer = SummaryWriter(config.training.log_dir)
        
        # Initialize tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path(self.config.training.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
    def save_checkpoint(self, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.training.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'gnn_state_dict': self.gnn.state_dict(),
            'diffusion_state_dict': self.diffusion.state_dict(),
            'gnn_optimizer': self.gnn_optimizer.state_dict(),
            'diffusion_optimizer': self.diffusion_optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint if needed
        if is_best:
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logging.info(f'Saved new best model with validation loss: {val_loss:.4f}')
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.current_epoch = checkpoint['epoch']
        self.gnn.load_state_dict(checkpoint['gnn_state_dict'])
        self.diffusion.load_state_dict(checkpoint['diffusion_state_dict'])
        self.gnn_optimizer.load_state_dict(checkpoint['gnn_optimizer'])
        self.diffusion_optimizer.load_state_dict(checkpoint['diffusion_optimizer'])
        
        logging.info(f'Loaded checkpoint from epoch {self.current_epoch}')
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.gnn.train()
        self.diffusion.train()
        
        total_gnn_loss = 0
        total_diffusion_loss = 0
        num_batches = 0
        
        train_loader = self.data_module.train_dataloader()
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Train GNN
            self.gnn_optimizer.zero_grad()
            _, gnn_loss = self.gnn.training_step(batch)
            gnn_loss = torch.tensor(gnn_loss, device=self.device)
            gnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), self.config.training.grad_clip)
            self.gnn_optimizer.step()
            
            # Train Diffusion
            self.diffusion_optimizer.zero_grad()
            _, diffusion_loss = self.diffusion.training_step(batch.x)
            diffusion_loss = torch.tensor(diffusion_loss, device=self.device)
            diffusion_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.diffusion.parameters(), self.config.training.grad_clip)
            self.diffusion_optimizer.step()
            
            # Update metrics
            total_gnn_loss += gnn_loss.item()
            total_diffusion_loss += diffusion_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'GNN Loss': f'{gnn_loss.item():.4f}',
                'Diff Loss': f'{diffusion_loss.item():.4f}'
            })
            
        return {
            'gnn_loss': total_gnn_loss / num_batches,
            'diffusion_loss': total_diffusion_loss / num_batches
        }
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate models."""
        self.gnn.eval()
        self.diffusion.eval()
        
        total_gnn_loss = 0
        total_diffusion_loss = 0
        num_batches = 0
        
        val_loader = self.data_module.val_dataloader()
        
        for batch in val_loader:
            batch = batch.to(self.device)
            
            # Evaluate GNN
            _, gnn_loss = self.gnn.training_step(batch)
            
            # Evaluate Diffusion
            _, diffusion_loss = self.diffusion.training_step(batch.x)
            
            total_gnn_loss += gnn_loss
            total_diffusion_loss += diffusion_loss
            num_batches += 1
            
        return {
            'val_gnn_loss': total_gnn_loss / num_batches,
            'val_diffusion_loss': total_diffusion_loss / num_batches
        }
        
    def train(self):
        """Main training loop."""
        logging.info("Starting training...")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self.train_epoch()
            
            # Validation phase
            val_metrics = self.validate()
            
            # Log metrics
            for name, value in {**train_metrics, **val_metrics}.items():
                self.writer.add_scalar(name, value, epoch)
                
            # Save checkpoint
            if epoch % self.config.training.save_interval == 0:
                self.save_checkpoint(val_metrics['val_gnn_loss'])
                
            # Early stopping check
            if val_metrics['val_gnn_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_gnn_loss']
                self.patience_counter = 0
                self.save_checkpoint(val_metrics['val_gnn_loss'], is_best=True)
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.training.early_stopping_patience:
                logging.info("Early stopping triggered!")
                break
                
            logging.info(
                f"Epoch {epoch} - Train GNN Loss: {train_metrics['gnn_loss']:.4f}, "
                f"Train Diff Loss: {train_metrics['diffusion_loss']:.4f}, "
                f"Val GNN Loss: {val_metrics['val_gnn_loss']:.4f}, "
                f"Val Diff Loss: {val_metrics['val_diffusion_loss']:.4f}"
            )
            
        logging.info("Training completed!")
        self.writer.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train ChronoHelix models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    config = ChronoHelixConfig.load(args.config)
    
    # Initialize and run trainer
    trainer = Trainer(config)
    trainer.train()