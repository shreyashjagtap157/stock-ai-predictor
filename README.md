# Stock AI Predictor (lightweight)

A compact, modular stock market prediction and crash-adaptation system optimized for moderate laptop hardware (Intel i5-9300H, GTX 1650 4GB, 24GB RAM).

Key design choices:
- Small LSTM model (PyTorch) with CPU-first defaults and optional GPU inference.
- Incremental learning (lightweight updates) during the day; full batch retrain after market close.
- Minimize GPU memory usage and batch data processing.
- Local storage of all fetched data; configurable API keys for Alpha Vantage / Finnhub.

Project layout:
- `src/data` - data fetchers
- `src/features` - feature engineering
- `src/models` - model code (small LSTM)
- `src/utils` - storage and helpers
- `src/cli` - command-line interface
- `docs` - installation and hardware tuning guidance

See `docs/INSTALL.md` for hardware-specific instructions and scheduling tips.
