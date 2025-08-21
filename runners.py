from collections import defaultdict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import get_batch
from tqdm import tqdm

class Runner: 
    """A class to handle training, evaluation, and generation for transformer models."""
    def __init__(self, net: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: str, metric_freq = 100):
        """Initialize the runner with model, loss function, optimizer and device.
        
        Args:
            net: The transformer model
            loss_fn: Loss function to optimize
            optimizer: Optimizer to use for training
            device: Device to run on ('cuda' or 'cpu')
            metric_freq: How often to compute and log metrics
        """
        self.net = net
        self.block_size = self.net.block_size
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.metric_freq = metric_freq
        self.metrics = defaultdict(list)

    def train(self, train_data, val_data, batch_size, iters = 1000):
        """Train the model on training data and validate periodically.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset  
            batch_size: Batch size for training
            iters: Number of training iterations
        """
        progress_bar = tqdm(range(iters), desc="Training")
        for i in progress_bar:
            # Get random batch
            x, y = get_batch(train_data, self.block_size, batch_size)

            # Forward and backward pass
            self.optimizer.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            out = self.net(x)
            loss = self.loss_fn(out.view(-1, out.shape[-1]), y.view(-1))
            acc = (torch.argmax(out.view(-1, out.shape[-1]), dim=1) == y.view(-1)).float().mean()

            loss.backward()
            self.optimizer.step()

            # Log metrics periodically
            if i % self.metric_freq == 0: 
                self.metrics["train_loss"].append(loss.item())
                self.metrics["train_acc"].append(acc.item())
                self.metrics["train_iter"].append(i)

                val_loss, val_acc = self.evaluate(val_data, batch_size)
                self.metrics["val_loss"].append(val_loss)
                self.metrics["val_acc"].append(val_acc)
                self.metrics["val_iter"].append(i)

                progress_bar.set_description(f"Train Loss: {loss.item():.4f}, Train Acc: {acc.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def evaluate(self, val_data, batch_size = 500):
        """Evaluate model on validation data.
        
        Args:
            val_data: Validation dataset
            batch_size: Batch size for validation
            
        Returns:
            Tuple of (validation loss, validation accuracy)
        """
        x, y = get_batch(val_data, self.block_size, batch_size)
        x, y = x.to(self.device), y.to(self.device)
        out = self.net(x)
        loss = self.loss_fn(out.view(-1, out.shape[-1]), y.view(-1))
        acc = (torch.argmax(out.view(-1, out.shape[-1]), dim=1) == y.view(-1)).float().mean()
        return loss.item(), acc.item()

    def generate(self, string_input, encode, decode, max_new_tokens = 50):
        """Generate new tokens given an input string.
        
        Args:
            string_input: Input string to continue generating from
            encode: Function to encode string to token indices
            decode: Function to decode token indices to string
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated string continuation
        """
        # Encode input and pad if needed
        sequence = encode(string_input)
        if len(sequence) < self.block_size:
            sequence = [0] * (self.block_size - len(sequence)) + sequence

        input_length = len(sequence)
        self.net.eval()
        for _ in range(max_new_tokens):
            # Get model prediction for next token
            x = torch.tensor(sequence[-self.block_size:], dtype=torch.long, device=self.device)
            x = x.unsqueeze(0)
            out = self.net(x)
            out = out[:, -1, :]
            probs = torch.softmax(out, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            sequence.append(next_idx.item())

        return decode(sequence[input_length:])

    def save(self, path):
        """Save model state dict to path."""
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        """Load model state dict from path."""
        self.net.load_state_dict(torch.load(path))

    def plot_metrics(self):
        """Plot training and validation metrics."""
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics["train_iter"], self.metrics["train_loss"], label="Train Loss")
        plt.plot(self.metrics["val_iter"], self.metrics["val_loss"], label="Validation Loss")
        plt.legend()

        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(self.metrics["train_iter"], self.metrics["train_acc"], label="Train Accuracy")
        plt.plot(self.metrics["val_iter"], self.metrics["val_acc"], label="Validation Accuracy")
        plt.legend()
        plt.show()
