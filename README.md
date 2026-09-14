# ASHRAE Energy Forecasting

This project forecasts hourly building electricity consumption using ASHRAE meter and weather data.

## Contents

- `code/task1.ipynb`: Main notebook covering data inspection, Building 105 selection and exploration, weather integration, feature engineering, XGBoost forecasting, evaluation, and comparison with a previous-week baseline.
- `code/building selection tool.ipynb`: Supporting branch for selecting reliable buildings. It filters complete meter-0 series, removes buildings with zero readings, and interactively visualizes the resulting 150 candidates.
- `data/`: Training meter data, weather data, building metadata, and the prepared Building 105 dataset.
- `reports/internship_progress_timeline.md`: Timeline of the project work and findings.
