# EMAformer reproduction

Main forecast matrix: 12 datasets, 4 forecast horizons and 6 seeds, 288 EMAformer runs. See the [Chinese reproduction report](%E5%A4%8D%E7%8E%B0%E6%8A%A5%E5%91%8A.md) for commands, the numerical comparison and evidence limits. The final cloud aggregation has been downloaded, and its six-seed means and population/sample standard deviations have been independently recomputed locally. Complete training artifacts remain locally verified for four runs. The summary and first export volume have been downloaded and checked; the full artifact export and checkpoint verification remain incomplete.

The progress statement above records the local-summary review dated 2026-11-26. Archive availability and verification are tracked separately; see the [archive recovery guide](docs/archive-recovery.md) for Release access, manifests, SHA-256 checks and restoration.

- Final six-seed tables from the downloaded aggregate include [Table 1](results-summary/Table1_EMAformer_horizon_average.csv), [Table 5](results-summary/Table5_EMAformer_per_horizon.csv), and the [release/best checkpoint comparison](results-summary/release_vs_best.csv).
- The [per-seed measurements](results-summary/main288_per_seed_metrics.csv) cover all 288 main runs and both checkpoint roles, with 576 rows of MSE, MAE, and RMSE. They come from the verified final aggregate. Download the training datasets from the original sources recorded in the dataset manifest.
- [Observed Table 1 summary](results-summary/table1-release-observed.csv): historical values transcribed from the cloud summary, rounded to five decimal places; checked against the downloaded final aggregate and unchanged at this precision. This CSV retains its original transcription provenance.
- [Run matrix](reproduction/manifest_paper_owned.json): use the multivariate_forecasting group for the 288 main runs.
- [Dataset manifest](datasets/manifest.json): public source links and local verification hashes.
- [Dependency lock](datasets/wheelhouse_linux/requirements-linux-cp310.lock): Linux x86_64, CPython 3.10, PyTorch 2.0.0 with CUDA 11.7.
- [Upstream code](EMAformer/README.md): pinned to b8f37c0c7d089e7263bbaeb681672a94f5afe233, with the [original MIT license](UPSTREAM-LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
