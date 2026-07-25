#!/bin/bash
# RedEdit: One-click environment setup
# Usage: bash setup_env.sh

set -e

echo "============================================"
echo "RedEdit Environment Setup"
echo "============================================"

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Python version: $PYTHON_VERSION"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "✓ Python >= 3.9 satisfied"
else
    echo "✗ Python >= 3.9 required. Current: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment (optional)
if [ "$1" = "--venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv rededit_env
    source rededit_env/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install PyTorch (CPU-only by default, change for GPU)
echo ""
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || \
    pip install torch torchvision

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"

python3 -c "
import sys
modules = ['numpy', 'PIL', 'torch', 'torchvision', 'openai', 'skimage', 'lpips', 'transformers', 'qwen_agent', 'tqdm']
ok = True
for m in modules:
    try:
        __import__(m)
        print(f'  ✓ {m}')
    except ImportError:
        print(f'  ✗ {m} (missing)')
        ok = False
if ok:
    print()
    print('All core dependencies installed successfully!')
else:
    print()
    print('Some dependencies are missing. Please check the error messages above.')
" || true

# API Key reminder
echo ""
echo "============================================"
echo "Next Steps"
echo "============================================"
echo ""
echo "1. Set your SiliconFlow API key:"
echo "   export SILICONFLOW_API_KEY=your_api_key_here"
echo "   (Get a key from https://cloud.siliconflow.cn/account/ak)"
echo ""
echo "2. Run a quick test:"
echo "   python run_mcts.py --image path/to/your/image.png"
echo ""
echo "3. For batch attacks:"
echo "   python run_mcts.py --dataset your_dataset.jsonl"
echo ""
echo "============================================"
