# Installation & Hardware Guidance (tailored for i5-9300H + GTX1650 + 24GB RAM)

1) Setup
- Create a Python 3.9+ virtual environment and install packages from `requirements.txt`.
- Install PyTorch separately from https://pytorch.org. For low GPU memory use the CPU build or a cuDNN build that fits your 4GB GPU. The CPU-only PyTorch is safest.

2) Minimize GPU memory usage
- Default config uses `device: cpu` in `config.yaml`.
- If you enable GPU, keep `batch_size` and `hidden_size` small (`batch_size <= 64`, `hidden_size <= 64`).
- Avoid training on GPU during trading hours. Use `incremental_epochs` small (1-3) for in-session updates.

3) Scheduling heavy tasks
- Schedule `retrain_full` after-market using Windows Task Scheduler or a cron job while the laptop is charging.
- Use `apscheduler` or system scheduler to run the job at `training.after_market_retrain_hour` configured in `config.yaml`.

4) Data and Storage
- Data is stored locally under `./data` by default (change in `config.yaml`). Parquet is preferred for space/time efficiency.

5) Resource tips
- Close other GPU-heavy apps when running batch retrain.
- Use incremental updates (few epochs) for quick intra-day adaptation.
- If you have a swap file, ensure you have enough free disk to avoid OOM.

6) Security: keep API keys out of source control. Put them in `config.yaml` which you do not commit.

