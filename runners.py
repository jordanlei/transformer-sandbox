from collections import defaultdict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import get_batch
from tqdm import tqdm

def generate(net, string_input, encode, decode, max_new_tokens=50, block_size=None):
    """Generate new tokens given an input string.
    Args:
        net: The transformer model
        string_input: Input string to continue generating from
        encode: Function to encode string to token indices
        decode: Function to decode token indices to string
        max_new_tokens: Maximum number of new tokens to generate
        block_size: Context window size (defaults to net.block_size)
        
    Returns:
        Generated string continuation
    """
    if block_size is None:
        block_size = net.block_size
        
    # Encode input and pad if needed
    sequence = encode(string_input)
    if len(sequence) < block_size:
        sequence = [0] * (block_size - len(sequence)) + sequence

    input_length = len(sequence)
    net.eval()
    
    for _ in range(max_new_tokens):
        # Get model prediction for next token
        x = torch.tensor(sequence[-block_size:], dtype=torch.long, device=next(net.parameters()).device)
        x = x.unsqueeze(0)
        out = net(x)
        out = out[:, -1, :]
        probs = torch.softmax(out, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        sequence.append(next_idx.item())

    return decode(sequence[input_length:])


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

    def train(self, train_data, val_data, batch_size, iters=1000, hook_fn=None):
        """Train the model on training data and validate periodically.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset  
            batch_size: Batch size for training
            iters: Number of training iterations
            hook_fn: Optional function called at each metric logging step
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

                # Call hook function if provided
                if hook_fn is not None:
                    hook_fn(i, self.metrics, self.net, self.optimizer)

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
        return generate(self.net, string_input, encode, decode, max_new_tokens, self.block_size)

    def save(self, path, metadata=None):
        """Save model state and architecture to path using the model's save method.
        
        Args:
            path: Path to save the model to
            metadata: Optional metadata to include in the saved file
        """
        self.net.save(path, metadata=metadata)

    @classmethod
    def load(cls, path, device='cpu', strict=True):
        """Load a model from path and return a new Runner instance.
        
        Args:
            path: Path to the saved model file
            device: Device to run on ('cuda', 'mps', or 'cpu')
            strict: Whether to strictly enforce state dict loading
            
        Returns:
            A new Runner instance with the loaded network, but None for optimizer and loss_fn
            
        Note:
            Assumes the model class has a load() class method.
        """
        # Ensure proper file extension
        if not path.endswith(('.pt', '.pth')):
            path += '.pt'
        
        try:
            # Load the saved dictionary to get model class info
            save_dict = torch.load(path, map_location='cpu')
            
            # Get the model class name and module
            model_class_name = save_dict.get('model_class')
            model_module = save_dict.get('model_module', 'networks')
            
            if not model_class_name:
                raise ValueError("Model file missing 'model_class' information")
            
            # Import the model class
            if model_module == 'networks':
                from networks import Transformer
                model_class = Transformer
            else:
                raise ImportError(f"Module {model_module} not supported yet")
            
            # Use the model class's load method directly
            new_net = model_class.load(path, device=device, strict=strict)
            
            # Create and return the new runner
            return cls(net=new_net, loss_fn=None, optimizer=None, device=device)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")
    
    @classmethod
    def load_with_metadata(cls, path, device='cpu'):
        """Load a model and return both the runner and metadata.
        
        Args:
            path: Path to the saved model file
            device: Device to run on
            
        Returns:
            Tuple of (runner, metadata) where metadata contains training info
        """
        # Ensure proper file extension
        if not path.endswith(('.pt', '.pth')):
            path += '.pt'
        
        try:
            save_dict = torch.load(path, map_location='cpu')
            
            if 'metadata' not in save_dict:
                print("Warning: No metadata found in model file")
                metadata = {}
            else:
                metadata = save_dict['metadata']
            
            # Load the runner normally
            runner = cls.load(path, device=device)
            
            return runner, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model with metadata from {path}: {e}")

    def plot_metrics(self):
        """Plot training and validation metrics."""
        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics["train_iter"], self.metrics["train_loss"], label="Train Loss")
        plt.plot(self.metrics["val_iter"], self.metrics["val_loss"], label="Validation Loss")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics["train_iter"], self.metrics["train_acc"], label="Train Accuracy")
        plt.plot(self.metrics["val_iter"], self.metrics["val_acc"], label="Validation Accuracy")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")

    
