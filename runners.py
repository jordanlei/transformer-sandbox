from collections import defaultdict
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import get_batch
from tqdm import tqdm
from collections import Counter
from networks import Transformer

def generate(net, string_input, tokenizer, detokenizer, max_new_tokens=50, block_size=None):
    """Generate new tokens given an input string.
    Args:
        net: The transformer model
        string_input: Input string to continue generating from
        tokenizer: Function to encode string to token indices
        detokenizer: Function to decode token indices to string
        max_new_tokens: Maximum number of new tokens to generate
        block_size: Context window size (defaults to net.block_size)
        
    Returns:
        Generated string continuation
    """
    if block_size is None:
        block_size = net.block_size
        
    # Encode input and pad if needed
    sequence = tokenizer(string_input)
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

    return detokenizer(sequence[input_length:])


class Runner: 
    """A class to handle training, evaluation, and generation for transformer models."""
    def __init__(self, net: nn.Module, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: str, metric_freq = 200):
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

    def generate(self, string_input, tokenizer, detokenizer, max_new_tokens = 50):
        """Generate new tokens given an input string.
        
        Args:
            string_input: Input string to continue generating from
            tokenizer: Function to encode string to token indices
            detokenizer: Function to decode token indices to string
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Generated string continuation
        """
        return generate(self.net, string_input, tokenizer, detokenizer, max_new_tokens, self.block_size)

    def save(self, path):
        """Save model state and architecture to path using the model's save method.
        
        Args:
            path: Path to save the model to
            metadata: Optional metadata to include in the saved file
        """
        metadata = {
            "model_class": self.net.__class__.__name__,
            "model_module": self.net.__module__,
            "block_size": self.block_size,
            "device": self.device,
            "metric_freq": self.metric_freq,
            "metrics": self.metrics,
            "loss_fn": self.loss_fn,
            "optimizer": self.optimizer,
        }
        self.net.save(path, metadata=metadata)

    @classmethod
    def load(cls, path, tokenizer, detokenizer, device='cpu', strict=True):
        """Load a model from path and return a new Runner instance.
        
        Args:
            path: Path to the saved model file
            device: Device to run on ('cuda', 'mps', or 'cpu')
            strict: Whether to strictly enforce state dict loading
            tokenizer: Function to encode string to token indices (required)
            detokenizer: Function to decode token indices to string (required)
            
        Returns:
            A new Runner instance with the loaded network, but None for optimizer and loss_fn
            
        Note:
            Assumes the model class has a load() class method.
        """
        try:
            # Load with weights_only=False to avoid PyTorch 2.6 compatibility issues
            # This is safe since we're loading our own saved models
            save_dict = torch.load(path, map_location='cpu', weights_only=False)
            net = Transformer.load(path, device=device, strict=strict)
            # Create and return the new runner (without tokenizer/detokenizer in constructor)
            # Access metadata from the saved dictionary
            metadata = save_dict.get('metadata', {})
            new_runner = cls(net=net, loss_fn=metadata.get('loss_fn'), optimizer=metadata.get('optimizer'), device=device)
            new_runner.metrics = metadata.get('metrics', {})
            return new_runner
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")
    
    def plot_metrics(self):
        """Plot basic training and validation metrics."""
        if not self.metrics or not self.metrics.get("train_iter"):
            print("No training metrics available to plot.")
            return
            
        # Plot loss
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics["train_iter"], self.metrics["train_loss"], label="Train Loss", color='blue')
        plt.plot(self.metrics["val_iter"], self.metrics["val_loss"], label="Validation Loss", color='red')
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.grid(True, alpha=0.3)

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics["train_iter"], self.metrics["train_acc"], label="Train Accuracy", color='blue')
        plt.plot(self.metrics["val_iter"], self.metrics["val_acc"], label="Validation Accuracy", color='red')
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_progress(self, tokenizer=None, detokenizer=None, prompt=None, max_new_tokens=50, save_path=None):
        """Plot comprehensive training progress including metrics and text generation samples.
        
        Args:
            tokenizer: Function to encode string to token indices (optional)
            detokenizer: Function to decode token indices to string (optional)
            prompt: Text prompt for generation sample (optional)
            max_new_tokens: Maximum number of new tokens to generate (default: 50)
            save_path: Optional path to save the figure (default: None, shows plot)
        """
        if not self.metrics or not self.metrics.get("train_iter"):
            print("No training metrics available to plot.")
            return
            
        # Create comprehensive figure
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        
        # Plot loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.metrics['train_iter'], self.metrics['train_loss'], label='Train Loss', color='blue', linewidth=2)
        ax1.plot(self.metrics['val_iter'], self.metrics['val_loss'], label='Validation Loss', color='red', linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy curves
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.metrics['train_iter'], self.metrics['train_acc'], label='Train Accuracy', color='blue', linewidth=2)
        ax2.plot(self.metrics['val_iter'], self.metrics['val_acc'], label='Validation Accuracy', color='red', linewidth=2)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Text generation sample (if tokenizer/detokenizer provided)
        if tokenizer and detokenizer:
            ax3 = fig.add_subplot(gs[1, :])
            ax3.axis('off')
            
            # Use provided prompt or default
            if prompt is None:
                prompt = "To be, or not to be, that is the question:"
            
            try:
                generated_text = self.generate(prompt, tokenizer, detokenizer, max_new_tokens)
                
                # Display prompt and generated text
                ax3.text(0.05, 0.95, "Prompt:", transform=ax3.transAxes, fontsize=12,
                        verticalalignment='top', fontfamily='monospace', fontweight='bold', color='blue')
                ax3.text(0.05, 0.90, prompt, transform=ax3.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace', color='blue')
                
                ax3.text(0.05, 0.75, "Generated Text:", transform=ax3.transAxes, fontsize=12,
                        verticalalignment='top', fontfamily='monospace', fontweight='bold', color='black')
                
                # Format and display generated text
                y_pos = 0.70
                for line in generated_text.split('\n'):
                    if y_pos < 0.05:
                        break
                    line = (line[:80] + "...") if len(line) > 80 else line.ljust(80)
                    ax3.text(0.05, y_pos, line, transform=ax3.transAxes, fontsize=9,
                            verticalalignment='top', fontfamily='monospace', color='black')
                    y_pos -= 0.04
                
                ax3.set_title(f'Text Generation Sample (max {max_new_tokens} tokens)', fontsize=14)
                
            except Exception as e:
                ax3.text(0.5, 0.5, f"Text generation failed:\n{str(e)}", 
                         transform=ax3.transAxes, ha='center', va='center', color='red')
                ax3.set_title('Text Generation Sample (Failed)', fontsize=14)
        else:
            # If no tokenizer, show training summary
            ax3 = fig.add_subplot(gs[1, :])
            ax3.axis('off')
            
            # Calculate training summary
            final_train_loss = self.metrics['train_loss'][-1] if self.metrics['train_loss'] else 'N/A'
            final_val_loss = self.metrics['val_loss'][-1] if self.metrics['val_loss'] else 'N/A'
            final_train_acc = self.metrics['train_acc'][-1] if self.metrics['train_acc'] else 'N/A'
            final_val_acc = self.metrics['val_acc'][-1] if self.metrics['val_acc'] else 'N/A'
            
            summary_text = f"""Training Summary:
            • Current Iteration: {self.metrics['train_iter'][-1] if self.metrics['train_iter'] else 'N/A'}
            • Current Train Loss: {final_train_loss:.4f if isinstance(final_train_loss, (int, float)) else final_train_loss}
            • Current Val Loss: {final_val_loss:.4f if isinstance(final_val_loss, (int, float)) else final_val_loss}
            • Current Train Acc: {final_train_acc:.4f if isinstance(final_train_acc, (int, float)) else final_train_acc}
            • Current Val Acc: {final_val_acc:.4f if isinstance(final_val_acc, (int, float)) else final_val_acc}
            • Model: {self.net.__class__.__name__}
            • Device: {self.device}"""
            
            ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace', color='black')
            ax3.set_title('Training Summary', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Progress plot saved to: {save_path}")
        else:
            plt.show()

    
