import torch
import glob
from PIL import Image
import re
from collections import defaultdict, Counter

def get_batch(data, block_size = 50, batch_size  = 64, masking = 0.0, mask_token = 1):
    # get a random starting index for each sample in our batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])

    # predict the next character
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # mask some of the tokens
    if masking > 0.0:
        mask = torch.rand(x.shape) < masking
        x[mask] = mask_token

    return x, y


def write_to_gif(dir = "temp_figures"):
    """Create GIF from saved figures with extended first and last frame durations."""
    print("Creating GIF from training progress figures...")
    try:
        # Get all PNG files in temp_figures directory
        image_files = sorted(glob.glob(f"{dir}/iteration_*.png"))
        
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


def load_shakespeare(file_path="shakespeare.txt", tokenize_type="word"):
    """Load and tokenize Shakespeare text, create vocabulary mappings.
    
    Args:
        file_path: Path to Shakespeare text file
        tokenize_type: Type of tokenization - 'word' or 'char'
        
    Returns:
        text: Raw text content
        vocab_size: Size of vocabulary 
        tokenize: Function to convert text to token indices
        detokenize: Function to convert token indices to text
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    if tokenize_type == "word":
        # Split into words, punctuation, newlines and tabs
        pattern = re.compile(r'\w+|[^\w\s]|\n|\t', re.UNICODE)
        tokens = pattern.findall(text)
        
        # Handle case sensitivity
        token_set = {t for t in tokens if t.islower()}
        tokens = [t.lower() if t.lower() in token_set else t for t in tokens]
        
        # Create vocabulary with special tokens
        vocab = ["<PAD>", "UNK"] + [w for w, _ in Counter(tokens).most_common(10000)]
        
        # Create mappings
        stoi = defaultdict(lambda: 1, {k:v for v,k in enumerate(vocab)})
        itos = defaultdict(lambda: "UNK", {v:k for k,v in stoi.items()})

        def tokenize(s):
            tokens = pattern.findall(s)
            return [stoi[t] if t in vocab else 
                   stoi[t.capitalize()] if t.capitalize() in vocab else
                   stoi[t.lower()] if t.lower() in vocab else
                   stoi["UNK"] 
                   for t in tokens]

        def detokenize(l):
            return re.sub(r' ([.,?!:;\n\t])', r'\1', " ".join(itos[i] for i in l))

    else:  # char tokenization
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)} 
        itos = {i: ch for i, ch in enumerate(chars)}
        
        def tokenize(s): return [stoi[c] for c in s]
        def detokenize(l): return "".join(itos[i] for i in l)

    return text, len(vocab) if tokenize_type == "word" else vocab_size, tokenize, detokenize
