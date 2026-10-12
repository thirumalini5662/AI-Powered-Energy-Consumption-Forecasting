# Project notes

## Intended use
Educational demonstration of hourly energy forecasting from simulated data.

## Model
RandomForestRegressor with calendar, lag-1, lag-24, and rolling 24-hour mean features.

## Validation
Chronological holdout. The app reports MAE, RMSE, and R² after training.

## Next improvements
- Add weather and holiday features.
- Compare with a seasonal-naive baseline.
- Test on a documented real dataset.
- Add prediction intervals and drift monitoring.
