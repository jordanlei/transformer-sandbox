import torch
import torch.nn as nn
from networks import Transformer
from runners import Runner, generate
from utils import get_batch
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob
from matplotlib import gridspec
from collections import Counter, defaultdict
import re
import argparse

if torch.backends.mps.is_available(): device = torch.device("mps")
elif torch.cuda.is_available(): device = torch.device("cuda")
else: device = torch.device("cpu")


def tokenize(text):
    # split on words, punctuation, newlines, and tabs, but exclude regular spaces
    # \S+ matches non-whitespace characters (words)
    # \n matches newlines
    # \t matches tabs
    # [^\w\s] matches any punctuation character
    pattern = re.compile(r'\w+|[^\w\s]|\n|\t', re.UNICODE)
    tokens = pattern.findall(text)
    # Create a set of lowercase tokens for faster lookup
    token_set = set()
    for t in tokens:
        if t.islower():
            token_set.add(t)
    # Process tokens in a list comprehension
    tokens = [t.lower() if t.lower() in token_set else t for t in tokens]
    return tokens

def load_shakespeare():
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenize(text)
    most_common = [w for w, c in Counter(tokens).most_common(10000)]
    vocab = ["<PAD>", "UNK"] + most_common
    stoi = defaultdict(lambda: 1, {k:v for v,k in enumerate(vocab)})
    itos = defaultdict(lambda: "UNK", {v:k for k,v in stoi.items()})

    def encode(s):
        tokens = tokenize(s)
        # Use list comprehension instead of append loop
        return [stoi[t] if t in vocab else 
                stoi[t.capitalize()] if t.capitalize() in vocab else
                stoi[t.lower()] if t.lower() in vocab else
                stoi["UNK"] 
                for t in tokens]

    def decode(l):
        # Use list comprehension for decoding, adding spaces between tokens
        return re.sub(r' ([.,?!:;\n\t])', r'\1', " ".join(itos[i] for i in l))

    return text, len(vocab), encode, decode

def write_to_gif():
    """Create GIF from saved figures with extended first and last frame durations."""
    print("Creating GIF from training progress figures...")
    try:
        # Get all PNG files in temp_figures directory
        image_files = sorted(glob.glob("temp_figures/iteration_*.png"))
        
        if image_files:
            # Open all images
            images = []
            for filename in image_files:
                img = Image.open(filename)
                images.append(img)
            
            # Create duration list: first and last frames stay longer
            durations = []
            for i in range(len(images)):
                if i == 0:  # First frame
                    durations.append(2000)  # 2 seconds
                elif i == len(images) - 1:  # Last frame
                    durations.append(3000)  # 3 seconds
                else:  # Middle frames
                    durations.append(500)   # 0.5 seconds
            
            # Save as GIF in main directory
            gif_filename = "animation.gif"
            images[0].save(
                gif_filename,
                save_all=True,
                append_images=images[1:],
                duration=durations,
                loop=0
            )
            print(f"GIF created successfully: {gif_filename}")
            print(f"Total frames: {len(images)}")
            print(f"First frame duration: {durations[0]}ms, Last frame duration: {durations[-1]}ms")
            
        else:
            print("No PNG files found to create GIF")
            
    except Exception as e:
        print(f"Error creating GIF: {e}")


def main():
    """
    Parse command line arguments and run training.
    """
    parser = argparse.ArgumentParser(description='Train Shakespeare transformer model')
    parser.add_argument('--iters', '-i', type=int, default=5000,
                       help='Number of training iterations (default: 5000)')
    parser.add_argument('--n_heads', type=int, default=3,
                       help='Number of attention heads (default: 10)')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of transformer layers (default: 10)')
    parser.add_argument('--embedding_size', type=int, default=128,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--block_size', type=int, default=80,
                       help='Context block size (default: 80)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Training batch size (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-4)')
    
    args = parser.parse_args()
    
    print(f"Training configuration:")
    print(f"  Iterations: {args.iters}")
    print(f"  Attention heads: {args.n_heads}")
    print(f"  Transformer layers: {args.n_layers}")
    print(f"  Embedding size: {args.embedding_size}")
    print(f"  Block size: {args.block_size}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    # Create temp_figures directory
    os.makedirs("temp_figures", exist_ok=True)
    
    text, vocab_size, encode, decode = load_shakespeare()
    data = torch.tensor(encode(text))
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    block_size = args.block_size
    batch_size = args.batch_size

    # Print an example of a training batch
    x, y = get_batch(train_data, block_size, batch_size)
    print("INPUT\n", "="*100, "\n", decode(x[1].tolist()))
    print("OUTPUT\n", "="*100, "\n", decode(y[1].tolist()))

    token_counts = Counter(train_data.tolist())
    token_counts = torch.tensor([token_counts[i] if i in token_counts else 0 for i in range(vocab_size)])
    token_counts = token_counts.clamp(min=1)  # Avoid division by zero
    weights = 1.0 / torch.log(token_counts + 1)
    weights = weights / weights.sum() * len(weights)  # Normalize
    weights = weights.to(device)

    net = Transformer(vocab_size, embedding_size=args.embedding_size, 
                     num_heads=args.n_heads, num_layers=args.n_layers, 
                     block_size=args.block_size, dropout=args.dropout).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    
    def training_hook(iteration, metrics, model, optimizer):
        """Save training progress figures showing loss curves, accuracy, and text generation."""

        # Print the current learning rate
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        # Apply learning rate decay
        if iteration > 0:
            # Decay learning rate exponentially every 1000 iterations
            if iteration % 1000 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9  # Reduce learning rate by 10%

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
        prompt = "Hark! the midnight bell doth toll, and shadows lengthen in the court.\n\
        Lo, a messenger comes, cloaked in haste, with tidings grave and unlooked for.\n\
        Methinks the stars do write upon the heavens the fate of mortal kings."
        try:
            generated_text = generate(model, prompt, encode, decode, max_new_tokens=200)
            
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
    
    runner = Runner(net, loss_fn, optimizer, device, metric_freq = 100)
    runner.train(train_data, val_data, batch_size=args.batch_size, iters=args.iters, hook_fn=training_hook)
    
    # Create GIF from saved figures
    write_to_gif()
    
    # Save the trained network using runner.save
    print("Saving the trained model...")
    try:
        runner.save("shakespeare_transformer_model")
        print("Model saved successfully as 'shakespeare_transformer_model'")
    except Exception as e:
        print(f"Error saving model: {e}")
        # Fallback: save just the network state dict
        try:
            torch.save(net.state_dict(), "shakespeare_transformer_state_dict.pt")
            print("Model state dict saved as 'shakespeare_transformer_state_dict.pt'")
        except Exception as e2:
            print(f"Error saving state dict: {e2}")
    
    # Clean up temp_figures directory
    import shutil
    shutil.rmtree("temp_figures")
    print("Cleaned up temp_figures directory")

if __name__ == "__main__":
    main()