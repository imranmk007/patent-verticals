# IP Visualization

pulls patent data from [HUPD](https://huggingface.co/datasets/HUPD/hupd) and makes some charts to show it's a good corpus for Mantis (diversity across IPC categories, accept/rejec label balance, text length distributions etc.)

## files

- `data-load.py` — streams patent JSONs from HuggingFace tarballs, caches to parquet
- `visualize.py` — reads the parquet, outputs figures + summary stats

## how to run

```bash
python data-load.py   # downloads + caches the data
python visualize.py   # generates figures/
```

outputs go to `figures/` and `stats/`
