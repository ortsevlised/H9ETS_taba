# Environment record

| Component | Version |
| --- | --- |
| OS | Windows 11 Home 10.0.26200 |
| GPU | NVIDIA GeForce RTX 5090 |
| GPU driver | 581.57 |
| Driver-reported CUDA | 13.0 |
| Python | 3.13.7 |
| PyTorch | 2.11.0+cu128 |
| TorchVision | 0.26.0+cu128 |
| PyTorch-bundled CUDA | 12.8 |
| timm | 1.0.28 |
| CodeCarbon | 3.3.0 |
| grad-cam (pytorch-grad-cam) | 1.5.5 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |

Environment is an isolated virtualenv at `.venv/` inside the project directory.

GPU sanity check: `torch.cuda.is_available()` is `True`, device capability reported as `(12, 0)` (Blackwell/RTX 50-series), and a CUDA matmul executed successfully.
