# SeqRep: Protein Sequence Representation

This project compares different deep learning models for protein sequence representation, focusing on **mLSTM (UniRep)** and **Transformer-based (ESM)** approaches.

## Project Structure

```
SeqRep/
├── esm/                    # ESM (Transformer-based) model
├── unirep/                 # UniRep (mLSTM) model
├── unirep_restructured/    # Restructured UniRep implementation
└── plot/                   # Visualization scripts
```

## Contributors

| Folder | Maintainer | Semester |
|--------|------------|----------|
| `esm/` | Zijie Zhao | Spring 2026 |
| `unirep/` | HsinYu Ko | Spring 2026 |
| `unirep_restructured/` | Zhaoshan Duan | Spring 2025 |

## Goals

- Compare ESM (Transformer) and UniRep (mLSTM) models for protein sequence representation
- Investigate reproducibility and determinism of model outputs
- Evaluate model performance and scalability

## References

- UniRep: https://github.com/churchlab/UniRep
- ESM: https://github.com/facebookresearch/esm
