# Paris Air Quality Forecasting

Forecast of hourly air pollution in Paris for the next 3 weeks (504 hours), for five pollutants: NO2, CO, O3, PM10 and PM25.

I compare simple baselines, a pretrained foundation model (Chronos-2, zero-shot) and a small LSTM, on the same validation window.

## Results

MAE on the last 504 hours of train, used as validation (lower is better):

| Model | NO2 | CO | O3 | PM10 | PM25 | Average |
|---|---|---|---|---|---|---|
| Chronos-2 joint | 6.348 | 0.037 | 12.790 | 5.292 | 3.374 | **5.568** |
| Chronos-2 independent | 7.714 | 0.048 | 12.916 | 5.418 | 3.260 | 5.871 |
| LSTM | 6.964 | 0.031 | 14.506 | 5.527 | 3.555 | 6.117 |
| Persistence | 7.429 | 0.046 | 16.160 | 5.622 | 4.237 | 6.699 |
| Seasonal naive 24h | 10.374 | 0.045 | 14.254 | 6.676 | 4.210 | 7.112 |
| Seasonal naive 168h | 9.469 | 0.044 | 17.303 | 6.439 | 3.555 | 7.362 |

Chronos-2 joint is the best on average and is used for the final submission. It is not the best on every pollutant: the LSTM wins on CO and the independent Chronos-2 on PM25.

The full report and analysis is in `report/report.md` 

## Repository

```
notebooks/01_forecasting.ipynb   analysis, models, evaluation and final forecast
report/report.md                 short report with the answers to the questions
data/raw/                        the Kaggle files (not in the repo)
outputs/                         submission.csv (not in the repo)
```

## Data

The data is not in the repository. Download `train.csv`, `test.csv` and `sample_submission.csv` from the Kaggle challenge of the course (https://www.kaggle.com/competitions/x-hec-ts-2025-predicting-air-quality-in-paris/leaderboard) and put them in `data/raw/`.

## How to run

**On Google Colab (recommended, this is how the results were produced)**

1. Open `notebooks/01_forecasting.ipynb` in Colab and choose a T4 GPU runtime.
2. Run all cells. The first cell installs the packages and asks you to upload the three CSV files.
3. The last cell downloads `submission.csv`.

**Locally**

```bash
uv sync
uv add torch chronos-forecasting
```

Then open the notebook with the `.venv` kernel. `uv sync` is enough for the data analysis and the baselines. The second line is needed for Chronos-2 and the LSTM.

## Setup used

- Chronos-2: `amazon/chronos-2`, chronos-forecasting 2.3.2, zero-shot, default context length, median forecast
- LSTM: 1 layer, 64 hidden units, 336 hours of input, trained for 15 epochs
- Hardware: Google Colab, Tesla T4

## Sources

- Ansari et al., "Chronos-2: From Univariate to Universal Forecasting", arXiv:2510.15821, 2025
- https://github.com/amazon-science/chronos-forecasting
