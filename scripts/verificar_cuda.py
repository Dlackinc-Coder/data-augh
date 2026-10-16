#!/usr/bin/env python3
"""Paso 4 del repo: prueba de fuego CUDA. Verifica que torch ve la RTX 4070 Ti."""
import torch

print(f"torch version: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version (runtime): {torch.version.cuda}")
    n = torch.cuda.device_count()
    print(f"GPUs detectadas: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        print(f"  [{i}] {props.name} — {total_gb:.1f} GB (CC {props.major}.{props.minor})")

    # Smoke test: una operación real en la GPU
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"Matmul en GPU OK — resultado shape={tuple(y.shape)}, mean={y.mean().item():.4f}")
    print("TODO CORRECTO: la GPU esta lista para entrenar.")
else:
    print("ERROR: CUDA no disponible. Revisa el driver (nvidia-smi).")
    raise SystemExit(1)
