# Internship Progress Timeline

| Date | Work completed |
| --- | --- |
| 18 August | Received the ASHRAE energy-consumption forecasting task and reviewed the dataset and requirements. |
| 19–20 August | Completed Step 1: filtered to electricity-meter data, checked data completeness and zero readings, and selected Building 105 because it has a complete year of non-zero hourly readings. |
| 21 August | Completed Step 2: explored Building 105's electricity use through time-series, hourly, weekday/weekend, distribution, and autocorrelation plots. Identified clear daily and weekly usage patterns. |
| 22 August | Completed Step 3: merged Building 105 with Site 1 weather data, handled missing weather observations, and assessed the relationship between air temperature and electricity consumption. |
| 24 August | Completed Step 4: engineered calendar and lag features (hour, day of week, month, 24-hour lag, and 168-hour lag) for forecasting. Also began Step 5 by considering an appropriate model for the time-series data. |
| 25 August | Completed Steps 5–8: selected and trained an XGBoost model using a chronological 80/20 split; evaluated predictions with RMSE and MAE; analysed errors around the winter-break drop in usage; and compared the model with the previous-week baseline. The baseline performed better, so documented possible improvements such as tuning and holiday features. |
