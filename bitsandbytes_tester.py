import bitsandbytes as bnb
import torch

print("PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA not detected by PyTorch.")

# Try to use bitsandbytes CUDA kernel
try:
    from bitsandbytes.cextension import COMPILED_WITH_CUDA
    print("bitsandbytes compiled with CUDA:", COMPILED_WITH_CUDA)
except Exception as e:
    print("❌ Could not check bitsandbytes CUDA support:", e)

# Optional: test a quantized linear layer
try:
    from bitsandbytes.nn import Linear4bit
    layer = Linear4bit(4, 4, bias=False)
    x = torch.randn(1, 4).to("cuda" if torch.cuda.is_available() else "cpu")
    y = layer(x)
    print("✅ bitsandbytes test passed! Output:", y)
except Exception as e:
    print("❌ bitsandbytes test failed:", e)
