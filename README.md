# EMAformer reproduction

Main forecast matrix: 12 datasets, 4 forecast horizons and 6 seeds, 288 EMAformer runs. See the [Chinese reproduction report](%E5%A4%8D%E7%8E%B0%E6%8A%A5%E5%91%8A.md) for commands, the numerical comparison and evidence limits. Main runs and six-seed aggregation were verified in the cloud. Four complete run artifacts were independently verified locally. The full cloud artifact export is pending.

- [Observed Table 1 summary](results-summary/table1-release-observed.csv): values transcribed from the cloud summary, rounded to five decimal places.
- [Run matrix](reproduction/manifest_paper_owned.json): use the multivariate_forecasting group for the 288 main runs.
- [Dataset manifest](datasets/manifest.json): public source links and local verification hashes.
- [Dependency lock](datasets/wheelhouse_linux/requirements-linux-cp310.lock): Linux x86_64, CPython 3.10, PyTorch 2.0.0 with CUDA 11.7.
- [Upstream code](EMAformer/README.md): pinned to b8f37c0c7d089e7263bbaeb681672a94f5afe233, with the [original license](UPSTREAM-LICENSE).
