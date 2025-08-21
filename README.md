# Transformer Sandbox

A Python project for experimenting with transformer models, featuring a simple implementation of the Transformer architecture and training utilities.

## 🎬 Training Visualization Demo

Watch the transformer learn in real-time! The training process generates an animated GIF showing the model's progress:

![Training Progress](animation.gif)

**What you'll see:**
- **Loss curves** (blue) and **validation loss** (red) decreasing over time
- **Accuracy curves** showing training and validation performance
- **Text samples** generated at each checkpoint using the prompt "HAMLET\n To be, or not to be?"

The GIF is automatically generated after training completes, showing the complete learning journey from random characters to Shakespeare-like patterns!

## Project Structure

```
transformer-sandbox/
├── demos/                          # Example notebooks and demonstrations
│   └── Demo Shakespeare.ipynb     # Shakespeare text generation demo
├── networks.py                     # Transformer model implementation
├── runners.py                      # Training and evaluation utilities
├── utils.py                        # Helper functions
├── test.py                         # Comprehensive test suite
├── shakespeare.txt                 # Shakespeare text dataset
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Features
- **Custom Transformer Implementation**: A PyTorch-based transformer model with configurable parameters
- **Training Runner**: Comprehensive training loop with metrics tracking and validation
- **Text Generation**: Inference capabilities for generating text continuations
- **🎭 Shakespeare Demo**: Complete, interactive tutorial for training transformers on literary text
  - Step-by-step character-level text generation
  - Comprehensive data preprocessing pipeline
  - Training visualization and metrics analysis
  - Real-time text generation with custom prompts
  - Optimized for Apple Silicon (MPS) and CPU

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd transformer-sandbox
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python test.py
```

## Usage

### 🎭 Shakespeare Demo - The Main Attraction

The centerpiece of this project is `demos/Demo Shakespeare.ipynb`, a comprehensive tutorial that demonstrates the full transformer training pipeline:

```bash
cd demos
jupyter notebook "Demo Shakespeare.ipynb"
```

#### What You'll Learn

This interactive notebook walks you through:

1. **Environment Setup**: Automatic device detection (MPS/CPU) and module imports
2. **Data Processing**: Loading Shakespeare text, creating vocabulary, and encoding/decoding functions
3. **Model Architecture**: Building a custom transformer with configurable parameters
4. **Training Process**: Complete training loop with real-time metrics and validation
5. **Visualization**: Training loss/accuracy plots for model analysis
6. **Text Generation**: Creating new Shakespeare-style text from custom prompts

#### Demo Features

- **Character-level generation** for fine-grained text control
- **Interactive training** with progress bars and real-time metrics
- **Customizable prompts** - try your own Shakespeare-style openings
- **Performance optimization** for Apple Silicon Macs
- **Educational explanations** for each step of the process

#### Sample Output

The demo generates text like:
```
Input: "OMELETTE. To fry, or not to fry?"
Output: "TOETHYOSUIW. Ss AXd po, Path phit ive? Y. TSNDHO..."
```

While not perfect English, it captures Shakespeare's linguistic patterns and structure!

### Using the Components Separately

#### Creating a Transformer Model

```python
from networks import Transformer

# Create a transformer with custom parameters
model = Transformer(
    vocab_size=1000,
    embedding_size=32,
    num_heads=3,
    num_layers=1,
    block_size=50,
    dropout=0.1
)
```

#### Training with the Runner

```python
from runners import Runner
import torch.nn as nn
import torch.optim as optim

# Setup training components
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
runner = Runner(model, loss_fn, optimizer, device="cpu")

# Train the model
runner.train(train_data, val_data, batch_size=64, iters=1000)
```

#### Text Generation

```python
# Generate text continuation
generated_text = runner.generate(
    string_input="To be or not to be",
    encode=encode_function,
    decode=decode_function,
    max_new_tokens=50
)
```

## Model Architecture

The transformer implementation includes:
- **Multi-head self-attention** with configurable number of heads
- **Position-wise feed-forward networks**
- **Layer normalization** and **residual connections**
- **Configurable embedding dimensions** and **number of layers**
- **Dropout** for regularization

## Requirements

- Python 3.7+
- PyTorch
- Matplotlib
- Jupyter (for running demos)

See `requirements.txt` for specific package versions.

## Contributing

Feel free to:
- Add new model architectures
- Create additional demos
- Improve the training utilities
- Add new datasets

## License

MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
