#!/usr/bin/env python3
"""
Minimal example showing how to load a saved model and generate text.
This demonstrates the simplest possible usage of the new Runner.load() functionality.
"""

from runners import Runner
from shakespeare_word import load_shakespeare

def main():
    # Load tokenizer
    _, _, encode, decode = load_shakespeare()
    
    # Load the saved model - that's it!
    loaded_runner = Runner.load("saved/shakespeare_transformer_model.pt")
    
    # Generate text
    prompt = "To be, or not to be, that is the question:"
    generated = loaded_runner.generate(prompt, encode, decode, max_new_tokens=50)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")

if __name__ == "__main__":
    main()
