# Computational environment

- Operating system: Windows
- Python: 3.12.10 (64-bit CPython)
- Reproduction environment: create a repository-local `.venv`; the author's working environment was stored outside the release tree and is not distributed
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB, compute capability 12.0
- NVIDIA driver observed during setup: 592.01
- PyTorch: 2.11.0+cu128
- CUDA runtime bundled with PyTorch: 12.8
- XGBoost: 3.3.0
- NumPy: 2.5.1
- pandas: 3.0.3
- SciPy: 1.18.0
- scikit-learn: 1.9.0
- Formal primary-model seeds: 20260722, 20260723, 20260724, 20260725, 20260726
- Formal learned-comparator and held-sport seeds: 20260722, 20260723, 20260724

GPU availability and a CUDA forward/backward tensor operation were tested successfully before model fitting. PyTorch CUDA 12.8 was chosen because the installed RTX 50-series GPU requires a Blackwell-capable build.

For a clean Windows environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The pinned file records the direct packages used for the study; it is not yet a hash-locked transitive environment export. A CPU-only installation can run tests, validators, and stored-result figure builds, but full model training was validated only on the GPU configuration above.
