$ErrorActionPreference = "Stop"

Write-Host "Setting Windows-friendly environment variables..."
$env:UV_LINK_MODE = "copy"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

Write-Host "Uninstalling existing PyTorch packages..."
uv pip uninstall torch torchvision torchaudio

Write-Host "Installing PyTorch nightly cu128 for RTX 5060 / sm_120..."
uv pip install -r requirements-5060-cu128.txt

Write-Host "Verifying CUDA kernel execution..."
uv --cache-dir .uv-cache run --no-sync python scripts/verify_cuda_5060.py
