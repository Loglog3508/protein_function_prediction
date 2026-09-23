# Repository Notes

- Use `E:\CUDA\envs\protein-gpu\python.exe` for the reproducible project environment. CUDA is available for PyTorch, but scikit-learn `SGDClassifier` remains CPU-only.
- Preserve the fixed seed-42 train/validation IDs under `artifacts/metrics/splits/seed42/` when comparing experiments.
- Screen label-wise hyperparameters with support-stratified labels through `src.sweep_sgd`; do not use only the first labels because they are frequency ordered.
- Use a new experiment ID and output prefix for every run. Never overwrite earlier metrics.
- Keep `artifacts/runs/`, submissions, `.npz`, and `.joblib` local. Commit configs, source, concise metrics, and experiment reports.
- CSV files are managed by Git LFS. Verify LFS status before pushing.
- Current handoff and second-iteration starting point are documented in `docs/iteration2_handoff.md`.
