import torch
import torch.nn as nn
from networks import Transformer
from runners import Runner, generate
from utils import get_batch, write_to_gif, load_shakespeare
import matplotlib.pyplot as plt
import os
from matplotlib import gridspec
if torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")


def main():
    # Create temp_figures directory
    os.makedirs("temp_figures", exist_ok=True)
    
    print("Loading data...")
    text, vocab_size, tokenize, detokenize = load_shakespeare(tokenize_type = "char")
    data = torch.tensor(tokenize(text))
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    block_size = 50
    batch_size = 64

    # Print an example of a training batch
    x, y = get_batch(train_data, block_size, batch_size)
    print("INPUT\n", "="*100, "\n", detokenize(x[1].tolist()))
    print("OUTPUT\n", "="*100, "\n", detokenize(y[1].tolist()))

    print("Creating model...")
    net = Transformer(vocab_size, embedding_size = 32, num_heads = 3, num_layers = 3, block_size = 50, dropout=0.1).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    
    def training_hook(iteration, metrics, model, optimizer):
        """Save training progress figures showing loss curves, accuracy, and text generation."""
        # Create figure layout
        fig = plt.figure(figsize=(10, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
        
        # Plot loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(metrics['train_iter'], metrics['train_loss'], label='Train Loss', color='blue')
        ax1.plot(metrics['val_iter'], metrics['val_loss'], label='Validation Loss', color='red')
        ax1.set_xlabel('Iteration', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.set_title('Training and Validation Loss', fontsize=16)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        
        # Plot accuracy curves
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(metrics['train_iter'], metrics['train_acc'], label='Train Accuracy', color='blue')
        ax2.plot(metrics['val_iter'], metrics['val_acc'], label='Validation Accuracy', color='red')
        ax2.set_xlabel('Iteration', fontsize=14)
        ax2.set_ylabel('Accuracy', fontsize=14)
        ax2.set_title('Training and Validation Accuracy', fontsize=16)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        
        # Plot text generation sample
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        for spine in ax3.spines.values():
            spine.set_visible(False)
        
        # Generate text sample
        prompt = "\nHAMLET\n To be, or not to be?"
        try:
            generated_text = runner.generate(prompt, tokenize, detokenize, max_new_tokens=200)
            
            # Display prompt
            ax3.text(0.05, 0.95, "Prompt:", transform=ax3.transAxes, fontsize=14,
                    verticalalignment='top', fontfamily='monospace', fontweight='bold', color='blue')
            ax3.text(0.05, 0.90, prompt, transform=ax3.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace', color='blue')
            
            # Display generated text
            ax3.text(0.05, 0.75, "Generated Text:", transform=ax3.transAxes, fontsize=14,
                    verticalalignment='top', fontfamily='monospace', fontweight='bold', color='black')
            
            # Format and display text lines
            y_pos = 0.70
            for line in generated_text.split('\n'):
                if y_pos < 0.05:
                    break
                padded_line = (line[:70] + "...") if len(line) > 70 else line.ljust(70)
                ax3.text(0.05, y_pos, padded_line, transform=ax3.transAxes, fontsize=12,
                        verticalalignment='top', fontfamily='monospace', color='black')
                y_pos -= 0.04
                
        except Exception as e:
            # Display error if text generation fails
            ax3.text(0.05, 0.75, "Error:", transform=ax3.transAxes, fontsize=14,
                    verticalalignment='top', fontfamily='monospace', fontweight='bold', color='red')
            ax3.text(0.05, 0.70, f"Text generation failed: {e}", transform=ax3.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace', color='red')
        
        ax3.set_title(f'Text Sample at Iteration {iteration}', fontsize=16)
        plt.tight_layout()
        
        # Save and cleanup
        filename = f"temp_figures/iteration_{iteration:06d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    print("Training model...")
    runner = Runner(net, loss_fn, optimizer, device, metric_freq = 100)
    runner.train(train_data, val_data, batch_size = 0, iters = 5000, hook_fn = training_hook)
    
    # Create GIF from saved figures
    write_to_gif()
    
    # Clean up temp_figures directory
    import shutil
    shutil.rmtree("temp_figures")
    print("Cleaned up temp_figures directory")

if __name__ == "__main__":
    main()