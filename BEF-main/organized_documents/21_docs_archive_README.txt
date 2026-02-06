Temporal Accumulators – Flash Drive Overview
===========================================

Start here:
-----------
1. paper/temporal_accumulators.pdf – 12-page note summarizing the threat model,
   the hybrid hash+algebraic accumulator, and how it connects to the Nov 4/6/11
   CS255 lectures (hash first, hash-only signatures, Merkle tries at scale).
2. paper/appendix_metrics.pdf – raw GPU vs CPU timing tables and algorithmic
   notes (kernel pseudocode, modular arithmetic tricks).
3. code/gpu_accumulator/ – the CUDA/PyTorch implementation, `demo_cuda()`
   notebook, and Colab instructions (HTML + zip bundle).
4. code/BEF_main/ – Rust BEF repo, ENN-C++ code, FusionAlpha, original notebooks.

Folder layout:
--------------
• paper/
    – temporal_accumulators.pdf (main note)
    – appendix_metrics.pdf (GPU kernels, tables)
• code/
    – gpu_accumulator/ (clean module, Colab-ready zip, HTML export)
    – BEF_main/      (full repo clone)
• docs/
    – README.txt (this file)
    – colab_instructions.txt

Quick Colab instructions:
-------------------------
1. Upload `code/gpu_accumulator/gpu_accumulator.zip` to your Colab runtime.
2. In a notebook cell:
       !unzip -o gpu_accumulator.zip
       !pip install -q ninja
3. Run the demo:
       import sys, gpu_accumulator
       sys.path.append('/content')
       from gpu_accumulator import demo_cuda
       demo_cuda(num_chunks=10, chunk_len=100_000)

Questions answered by the note:
-------------------------------
• Why a hybrid hash+sketch accumulator addresses the Nov 4 “hash-first” advice.
• How a lightweight algebraic sketch removes the Nov 11 Merkle-trie state cost.
• What the security/performance tradeoffs are vs pure Merkle tries.
• How to classify the construction (vector commitment / streaming PCS).

Supporting files:
-----------------
• `code/gpu_accumulator/batch_streaming_accum.html` – readable notebook export.
• `code/gpu_accumulator/benchmarks.csv` – raw CPU/GPU timing data (GTX 1650,
   A100).
• `code/gpu_accumulator/README.md` – module documentation.

