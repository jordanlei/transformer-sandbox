import torch
import torch.nn as nn
from networks import Transformer
from runners import Runner, generate
from utils import get_batch
import matplotlib.pyplot as plt
import os
from PIL import Image
import glob

if torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")

def load_shakespeare():
    with open("shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_char = {i: ch for i, ch in enumerate(chars)}
    def encode(s): return [char_to_index[c] for c in s]
    def decode(l): return "".join([index_to_char[i] for i in l])
    return text, vocab_size, encode, decode


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
    # Create temp_figures directory
    os.makedirs("temp_figures", exist_ok=True)
    
    text, vocab_size, encode, decode = load_shakespeare()
    data = torch.tensor(encode(text))
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    block_size = 50
    batch_size = 64

    # Print an example of a training batch
    x, y = get_batch(train_data, block_size, batch_size)
    print("INPUT\n", "="*100, "\n", decode(x[1].tolist()))
    print("OUTPUT\n", "="*100, "\n", decode(y[1].tolist()))

    net = Transformer(vocab_size, embedding_size = 32, num_heads = 3, num_layers = 1, block_size = 50, dropout=0.1).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    
    # Define training hook within main to access encode/decode functions
    def training_hook(iteration, metrics, model):
        """Save training progress figures instead of printing."""
        # Create a figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Loss curves
        ax1.plot(metrics['train_iter'], metrics['train_loss'], label='Train Loss', color='blue')
        ax1.plot(metrics['val_iter'], metrics['val_loss'], label='Validation Loss', color='red')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Accuracy curves
        ax2.plot(metrics['train_iter'], metrics['train_acc'], label='Train Accuracy', color='blue')
        ax2.plot(metrics['val_iter'], metrics['val_acc'], label='Validation Accuracy', color='red')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Text sample
        ax3.axis('off')
        ax3.set_xlim(0, 1)  # Fixed x-axis limits
        ax3.set_ylim(0, 1)  # Fixed y-axis limits
        
        prompt = "\nHAMLET\n To be, or not to be?"
        try:
            generated_text = generate(model, prompt, encode, decode, max_new_tokens=200)
            
            # Add prompt in blue
            ax3.text(0.05, 0.95, "Prompt:", transform=ax3.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace', fontweight='bold', color='blue')
            ax3.text(0.05, 0.90, prompt, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace', color='blue')
            
            # Add generated text in black - clip long lines
            ax3.text(0.05, 0.80, "Generated Text:", transform=ax3.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace', fontweight='bold', color='black')
            
            # Split text into lines and pad/clip to exactly 80 characters
            lines = generated_text.split('\n')
            y_pos = 0.75
            for line in lines:
                if y_pos < 0.05:  # Stop if we run out of vertical space
                    break
                # Pad or clip line to exactly 80 characters
                if len(line) > 80:
                    padded_line = line[:80] + "..."
                else:
                    padded_line = line.ljust(80)  # Pad with spaces to 80 chars
                
                ax3.text(0.05, y_pos, padded_line, transform=ax3.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace', color='black')
                y_pos -= 0.04  # Move down for next line
            
        except Exception as e:
            ax3.text(0.05, 0.95, "Prompt:", transform=ax3.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace', fontweight='bold', color='blue')
            ax3.text(0.05, 0.90, prompt, transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace', color='blue')
            ax3.text(0.05, 0.80, "Error:", transform=ax3.transAxes, fontsize=12,
                    verticalalignment='top', fontfamily='monospace', fontweight='bold', color='red')
            ax3.text(0.05, 0.75, f"Text generation failed: {e}", transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace', color='red')
        
        ax3.set_title(f'Text Sample at Iteration {iteration}')
        
        plt.tight_layout()
        
        # Save the figure
        filename = f"temp_figures/iteration_{iteration:06d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()  # Close to free memory
    
    runner = Runner(net, loss_fn, optimizer, device, metric_freq = 100)
    runner.train(train_data, val_data, batch_size = 500, iters = 5000, hook_fn = training_hook)
    
    # Create GIF from saved figures
    write_to_gif()
    
    # Clean up temp_figures directory
    import shutil
    shutil.rmtree("temp_figures")
    print("Cleaned up temp_figures directory")

if __name__ == "__main__":
    main()