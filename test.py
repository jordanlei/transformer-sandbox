#!/usr/bin/env python3
"""
Test script for Transformer Sandbox
Runs comprehensive tests to verify all components are working correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all required modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"  ❌ PyTorch import failed: {e}")
        return False
    
    try:
        import matplotlib
        print(f"  ✅ Matplotlib: {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import jupyter
        print(f"  ✅ Jupyter: imported successfully")
    except ImportError as e:
        print(f"  ❌ Jupyter import failed: {e}")
        return False
    
    try:
        from networks import Transformer
        print(f"  ✅ Transformer: imported successfully")
    except ImportError as e:
        print(f"  ❌ Transformer import failed: {e}")
        return False
    
    try:
        from runners import Runner
        print(f"  ✅ Runner: imported successfully")
    except ImportError as e:
        print(f"  ❌ Runner import failed: {e}")
        return False
    
    try:
        from utils import get_batch
        print(f"  ✅ Utils: imported successfully")
    except ImportError as e:
        print(f"  ❌ Utils import failed: {e}")
        return False
    
    print("  ✅ All imports successful!")
    return True

def test_device_detection():
    """Test device detection (MPS/CPU)."""
    print("\n🔍 Testing device detection...")
    
    try:
        import torch
        
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"  ✅ MPS (Apple Silicon GPU) available: {device}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"  ✅ CUDA available: {device}")
        else:
            device = torch.device("cpu")
            print(f"  ✅ Using CPU: {device}")
        
        return True
    except Exception as e:
        print(f"  ❌ Device detection failed: {e}")
        return False

def test_transformer_model():
    """Test that the Transformer model can be created and run."""
    print("\n🔍 Testing Transformer model...")
    
    try:
        import torch
        from networks import Transformer
        
        # Create a small test model
        vocab_size = 100
        embedding_size = 32
        num_heads = 2
        num_layers = 1
        block_size = 50
        
        model = Transformer(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            num_heads=num_heads,
            num_layers=num_layers,
            block_size=block_size
        )
        
        print(f"  ✅ Model created successfully")
        print(f"    - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        x = torch.randint(0, vocab_size, (batch_size, block_size))
        output = model(x)
        
        expected_shape = (batch_size, block_size, vocab_size)
        if output.shape == expected_shape:
            print(f"  ✅ Forward pass successful: {output.shape}")
        else:
            print(f"  ❌ Unexpected output shape: {output.shape}, expected {expected_shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Transformer test failed: {e}")
        traceback.print_exc()
        return False

def test_runner_class():
    """Test that the Runner class can be instantiated."""
    print("\n🔍 Testing Runner class...")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from networks import Transformer
        from runners import Runner
        
        # Create a small test model
        model = Transformer(
            vocab_size=100,
            embedding_size=32,
            num_heads=2,
            num_layers=1,
            block_size=50
        )
        
        # Create loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
        # Create runner
        runner = Runner(model, loss_fn, optimizer, 'cpu')
        print(f"  ✅ Runner created successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Runner test failed: {e}")
        traceback.print_exc()
        return False

def test_utils():
    """Test utility functions."""
    print("\n🔍 Testing utility functions...")
    
    try:
        import torch
        from utils import get_batch
        
        # Create test data
        data = torch.randint(0, 100, (1000,))
        block_size = 50
        batch_size = 4
        
        x, y = get_batch(data, block_size, batch_size)
        
        if x.shape == (batch_size, block_size) and y.shape == (batch_size, block_size):
            print(f"  ✅ get_batch successful: x={x.shape}, y={y.shape}")
        else:
            print(f"  ❌ Unexpected batch shapes: x={x.shape}, y={y.shape}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Utils test failed: {e}")
        traceback.print_exc()
        return False

def test_jupyter_functionality():
    """Test basic Jupyter functionality."""
    print("\n🔍 Testing Jupyter functionality...")
    
    try:
        import subprocess
        import sys
        
        # Test if jupyter command is available
        result = subprocess.run([sys.executable, '-m', 'jupyter', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"  ✅ Jupyter command available")
            print(f"    - Output: {result.stdout.strip()}")
        else:
            print(f"  ❌ Jupyter command failed: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Jupyter test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Transformer Sandbox Tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Device Detection", test_device_detection),
        ("Transformer Model", test_transformer_model),
        ("Runner Class", test_runner_class),
        ("Utility Functions", test_utils),
        ("Jupyter Functionality", test_jupyter_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"  ✅ {test_name} test passed")
            else:
                print(f"  ❌ {test_name} test failed")
        except Exception as e:
            print(f"  ❌ {test_name} test crashed: {e}")
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Transformer Sandbox is ready to use!")
        print("\n🚀 Next steps:")
        print("  1. cd demos")
        print("  2. jupyter notebook 'Demo Shakespeare.ipynb'")
        print("  3. Start experimenting with transformers!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

