import torch
import torch.nn as nn
from networks import Transformer
from runners import Runner, generate
from utils import get_batch, load_shakespeare, write_to_gif
import os
import argparse
from collections import Counter

# Set device based on availability
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description='Train Shakespeare transformer model')
    parser.add_argument('--iters', '-i', type=int, default=5000,
                       help='Number of training iterations')
    parser.add_argument('--n_heads', type=int, default=3,
                       help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=3,
                       help='Number of transformer layers')
    parser.add_argument('--embedding_size', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--block_size', type=int, default=150,
                       help='Context block size')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate')
    parser.add_argument('--tokenize_type', type=str, default="word",
                       help='Type of tokenization (word or char)')
    args = parser.parse_args()
    
    # Print training configuration
    print("Training configuration:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    
    # Setup
    os.makedirs("temp_figures", exist_ok=True)
    text, vocab_size, tokenize, detokenize = load_shakespeare(tokenize_type = args.tokenize_type)
    data = torch.tensor(tokenize(text))
    train_data, val_data = data[:int(0.9*len(data))], data[int(0.9*len(data)):]

    # Calculate token weights for loss function
    if args.tokenize_type == "word":
        token_counts = Counter(train_data.tolist())
        weights = torch.tensor([token_counts.get(i, 1) for i in range(vocab_size)]).clamp(min=1)
        weights = (1.0 / torch.log(weights + 1))
        weights = (weights / weights.sum() * len(weights)).to(device)
    else:
        weights = None

    # Initialize model, loss and optimizer
    net = Transformer(
        vocab_size, 
        embedding_size=args.embedding_size,
        num_heads=args.n_heads,
        num_layers=args.n_layers,
        block_size=args.block_size,
        dropout=args.dropout
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)

    # Define simple training hook that uses Runner's native plot_progress method
    def training_hook(iteration, metrics, model, optimizer):
        """Simple training hook that plots progress using Runner's native method."""
        # Use the runner's plot_progress method to save the current training state
        runner.plot_progress(tokenizer=tokenize, detokenizer=detokenize, 
                           prompt="MACBETH.\nI'll not be moved;\nDeprived of that which\nMakes us kinsmen.", 
                           max_new_tokens=100, 
                           save_path=f"temp_figures/iteration_{iteration:06d}.png")

    # Train model
    runner = Runner(net, loss_fn, optimizer, device, metric_freq=100)
    runner.train(train_data, val_data, batch_size=args.batch_size, iters=args.iters, hook_fn=training_hook)
    
    # Create training animation from saved progress plots
    print("Creating training animation...")
    write_to_gif()
    
    # Save model
    print("Saving the trained model...")
    runner.save("saved/shakespeare_transformer_model.pt")
    print("Model saved successfully as 'saved/shakespeare_transformer_model.pt'")
    
    # Test model loading
    try:
        print("Loading model using Runner.load()...")
        loaded_runner = Runner.load("saved/shakespeare_transformer_model.pt", 
                                  tokenizer=tokenize, 
                                  detokenizer=detokenize,
                                  device=device)
        print("✓ Model loaded successfully using Runner.load()")
        # Verify loaded model
        print(f"✓ Loaded runner.net type: {type(loaded_runner.net)}")
        print(f"✓ Loaded runner.optimizer: {loaded_runner.optimizer}")
        print(f"✓ Loaded runner.loss_fn: {loaded_runner.loss_fn}")
        print(f"✓ Loaded runner.device: {loaded_runner.device}")
        print(f"✓ Loaded runner.block_size: {loaded_runner.block_size}")

        # Test generation
        print("\nTesting text generation with loaded model...")
        test_prompt = "MACBETH.\nI'll not be moved;\nDeprived of that which\nMakes us kinsmen."
        generated_text = loaded_runner.generate(test_prompt, tokenize, detokenize, max_new_tokens=100)
        print(f"✓ Text generation successful!")
        print(f"Prompt: {test_prompt}")
        print(f"Generated: {generated_text}")
        
        # Test network access
        print("\nTesting direct network access...")
        loaded_net = loaded_runner.net
        print(f"✓ Network loaded: {type(loaded_net)}")
        print(f"✓ Network block_size: {loaded_net.block_size}")
        print(f"✓ Network vocab_size: {loaded_net.vocab_size}")
        
        print("\n🎉 All tests passed! Runner.load() is working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing Runner.load(): {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    import shutil
    shutil.rmtree("temp_figures")
    print("Cleaned up temp_figures directory")

if __name__ == "__main__":
    main()