# Paris air quality forecasting: short report

The goal is to forecast 504 hours (3 weeks) of five pollutants in Paris: NO2, CO, O3, PM10 and PM25. I compare two simple baselines, Chronos-2 used zero-shot, and a small LSTM. All the code and figures are in `notebooks/01_forecasting.ipynb`.

## 1. The data

Train is hourly, from 2020-01-01 to 2024-09-03 (40,991 rows). It is sorted, with no duplicated and no missing timestamp. Test is the 504 hours that follow.

**Why forecasting and not regression.** The rows are not independent. The value at one hour is very close to the value one hour before (autocorrelation at lag 1 is between 0.85 and 0.96), and the task is to predict the future using only the past.

**Why a random split is misleading.** With a random split, the hours just before and just after each test hour are in the training set. The model can almost copy its neighbours, so the score looks much better than it really is. It also does not test the real task, which is to predict three weeks with nothing known after the start.

**Patterns I see.**

- Daily: NO2 has two rush-hour peaks (about 29 around 6h to 7h and 27 around 19h to 20h, against 16 at 13h). O3 does the opposite (33 at 6h, about 67 at 14h).
- Weekly: NO2 is around 24 from Tuesday to Friday and 16.6 on Sunday.
- Yearly: O3 is about 68 in June and 30 in December.
- The autocorrelation has a bump every 24 hours. At 168 hours it is still 0.53 for O3 and 0.35 for NO2, but only about 0.15 for PM10 and PM25.
- Between pollutants: PM10 and PM25 move together (0.89), NO2 and CO too (0.71), and O3 goes against NO2 (-0.64).

**Missing values.** CO is missing 31% of the time and almost all of 2021. PM10 is missing 17%, in many short gaps. NO2 is missing 8%, with one outage of 1,500 hours. O3 and PM25 are nearly complete.

I do not interpolate. Interpolation needs the value after the gap, so a gap at the end of train would be filled with values from the validation or test period, which is a leak. Instead, the baselines use the last observed value, Chronos-2 receives the gaps as they are, and the LSTM gets values filled forward from the past plus a flag saying if the value was observed. Errors are computed on observed values only.

## 2. Validation and baselines

The validation set is the last 504 hours of train (2024-08-13 23:00 to 2024-09-03 22:00), the same length as the test period. Models only see what comes before.

**Why a 168-hour lag.** 168 hours is one week, so the forecast copies the same hour of the same weekday. This keeps the daily cycle and the difference between working days and weekends, which matters for traffic pollution. A 24-hour lag would copy a Sunday onto a Monday.

**Why MAE per pollutant.** The scales are very different (CO is around 0.2, O3 around 50), so the average mostly reflects O3. And the best model is not the same for each pollutant.

## 3. Foundation models

**What it is.** A time-series foundation model is trained once on a very large and varied collection of series, so it learns general patterns like seasonality and trend. It can then forecast a new series it has never seen. A model trained only on this dataset, like my LSTM, knows nothing except the Paris data.

**Zero-shot.** The weights of Chronos-2 are never updated on the Paris data. The model only receives the history as context and outputs the forecast.

**Forecasting all pollutants together.** Advantage: the pollutants are correlated, so one can help another, especially when one is missing. NO2 has no value in the two weeks before the split, and its error drops from 7.71 to 6.35 with the joint model. Risk: the noise and the gaps of one series can disturb the others, and the gain is not guaranteed. PM25 is slightly worse with the joint model (3.37 against 3.26).

**Types of variables.**

| Variable | Type |
|---|---|
| Pollutant history | Target |
| Historical weather | Past-only covariate |
| Hour of day | Known-future covariate |
| Future weather observations | Not usable. They do not exist when the forecast is made, so using them would be a leak. Only a weather forecast made before the period could be a known-future covariate. |

**Why the median.** The median is the value that minimises the expected absolute error, while the mean minimises the squared error. Pollution has rare high peaks, so the mean is pulled up and the median gives a lower MAE.

## 4. Setup

| | |
|---|---|
| Foundation model | `amazon/chronos-2`, chronos-forecasting 2.3.2, torch 2.11.0 |
| Hardware | Google Colab, Tesla T4 |
| Chronos-2 settings | default context (8,192 hours), horizon 504, median forecast, no fine-tuning, no covariates |
| LSTM | 1 layer, 64 hidden units, 336 hours of input, the 504 hours predicted in one pass, L1 loss, 15 epochs, Adam 1e-3, seed 0 |

## 5. Results

MAE on the validation window (lower is better):

| Model | NO2 | CO | O3 | PM10 | PM25 | Average |
|---|---|---|---|---|---|---|
| Chronos-2 joint | **6.348** | 0.037 | **12.790** | **5.292** | 3.374 | **5.568** |
| Chronos-2 independent | 7.714 | 0.048 | 12.916 | 5.418 | **3.260** | 5.871 |
| LSTM | 6.964 | **0.031** | 14.506 | 5.527 | 3.555 | 6.117 |
| Persistence | 7.429 | 0.046 | 16.160 | 5.622 | 4.237 | 6.699 |
| Seasonal naive 24h | 10.374 | 0.045 | 14.254 | 6.676 | 4.210 | 7.112 |
| Seasonal naive 168h | 9.469 | 0.044 | 17.303 | 6.439 | 3.555 | 7.362 |

**Which approach works best.** The joint Chronos-2 model is first on average, then the independent one, then the LSTM. All three beat the baselines. The joint model is better than the independent one mainly on NO2 and CO. NO2 has no recent values and CO has the largest gaps, so they probably gain the most from the other pollutants.

**Is the best model best everywhere?** No. Chronos-2 joint wins on NO2, O3 and PM10, but the LSTM wins on CO and the independent Chronos-2 on PM25. The average hides this because it is dominated by O3. Another surprise is that persistence beats the 168-hour seasonal naive on average, although seasonal naive is the better baseline on CO and PM25.

**Beginning and end of the horizon.** Average MAE per week:

| Model | Week 1 | Week 2 | Week 3 |
|---|---|---|---|
| Chronos-2 joint | 4.90 | 5.42 | 6.41 |
| LSTM | 5.96 | 6.00 | 6.52 |
| Persistence | 6.34 | 6.97 | 6.81 |

Chronos-2 gets worse as the horizon grows. Its lead over persistence goes from 1.4 in the first week to 0.4 in the third.

**Where Chronos-2 fails against seasonal naive.** Seasonal naive is better on PM10 in week 1 (4.07 against 4.64) and on PM25 in week 2 (2.34 against 2.91). More generally, Chronos-2 predicts a smooth average day and misses the peaks. The clearest failure is the PM25 episode around August 31: the observed values go above 25 and the forecast stays under 10. The clearest success is O3, where it follows the daily cycle for the whole three weeks.

## 6. Limitations

- **Data.** Large gaps: CO lost almost all of 2021, and NO2 has no value in the two weeks before the validation split. I also have no weather data, which drives pollution peaks.
- **Evaluation.** I use a single 3-week window at the end of August, so the ranking could change in another season. NO2 and PM10 are scored on only 394 and 299 observed hours out of 504. The final model is chosen on this same window.
- **Model.** Chronos-2 is used without fine-tuning and without covariates. The LSTM is trained once, with one seed and no tuning, and its score is not stable: the same code gave 5.68 on another machine against 6.12 on Colab. Both give smooth forecasts that miss peaks.

## 7. Final forecast

I use Chronos-2 joint, the model with the lowest average MAE on validation, with all of train as context. The submission has the same columns and the same 504 timestamps as the sample file, in the same order, with no missing value. These checks are asserts in the last cell of the notebook.

## Sources

- Ansari et al., "Chronos-2: From Univariate to Universal Forecasting", arXiv:2510.15821, 2025.
- Chronos-2 code and quick-start notebook: https://github.com/amazon-science/chronos-forecasting
- Data: Kaggle challenge of the course.

I used no external data. The only extra information is the calendar (hour, weekday, day of year) computed from the timestamps.
