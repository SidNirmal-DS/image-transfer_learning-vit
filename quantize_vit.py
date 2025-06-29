import torch
from transformers import ViTForImageClassification

# 1. Enable quantization engine
torch.backends.quantized.engine = "qnnpack"

# 2. Load pretrained ViT model
model = ViTForImageClassification.from_pretrained("vit-cifar10-checkpoint")

# 3. Apply dynamic quantization (only Linear layers)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 4. Convert to TorchScript
example_input = torch.rand(1, 3, 224, 224)
scripted_model = torch.jit.trace(quantized_model, example_input, strict=False)

# 5. Save as TorchScript
torch.jit.save(scripted_model, "vit_quantized_cifar10.pt")

print("✅ Quantized model saved as vit_quantized_cifar10.pt")