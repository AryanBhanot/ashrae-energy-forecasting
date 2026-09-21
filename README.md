# ASHRAE Energy Forecasting

This repository provides an end-to-end machine learning pipeline to forecast hourly building electricity consumption using the ASHRAE - Great Energy Predictor III dataset. It includes data extraction tools, exploratory time-series analysis, feature engineering (lags, weather integration, calendar features), gradient-boosted tree modeling (XGBoost), and deployment-ready web applications (Streamlit dashboard and Flask service) packaged for Docker and GCP Cloud Run.

---

## Repository Structure

```text
ashrae-energy-forecasting/
├── apps/                                 # Web applications and service entry points
│   ├── flask/                            # Flask web service
│   │   ├── static/                       # Static CSS assets
│   │   │   └── site.css
│   │   ├── templates/                    # Jinja2 HTML templates
│   │   │   └── hello_there.html
│   │   └── app.py                        # Flask server entry point
│   └── streamlit/                        # Interactive Streamlit dashboard
│       └── app.py                        # Streamlit application entry point
├── data/                                 # Datasets and metadata
│   ├── building/                         # Standardized building meter datasets
│   │   ├── 105/                          # Building 105 meter data (building_105.csv, meter_0.csv)
│   │   └── 1074/                         # Building 1074 meter data (building_1074.csv)
│   ├── building_metadata(in).csv         # Metadata for ASHRAE buildings (site, primary use, square feet, etc.)
│   ├── weather_train.csv                 # Hourly site weather observations (air temperature, dew point, etc.)
│   └── train.csv                         # Raw ASHRAE training observations (optional / local)
├── models/                               # Serialized machine learning models
│   └── building_1074_six_months_model.pkl# Trained XGBoost model artifact for Building 1074
├── notebooks/                            # Jupyter notebooks for data extraction, analysis, and ML
│   ├── building_selection_tool.ipynb     # Interactive tool to filter and identify reliable candidate buildings
│   ├── building_data_extraction.ipynb    # Utility to filter train.csv and export standardized building meter slices
│   ├── task1.ipynb                       # Full pipeline for Building 105: EDA, lag features, XGBoost & baseline
│   └── six_months.ipynb                  # 6-month forecasting analysis and modeling for Building 1074
├── reports/                              # Project reports and documentation deliverables
│   ├── Task1_Writeup.pdf                 # Comprehensive technical writeup and methodology report
│   └── internship_progress_timeline.md   # Chronological log of project milestones and tasks completed
├── .dockerignore                         # Docker build exclusion rules
├── .gitignore                            # Git ignore rules for repository hygiene
├── Dockerfile                            # Production container configuration (GCP Cloud Run ready)
├── requirements.txt                      # Project Python dependencies
└── README.md                             # Project overview and documentation
```

---

## Methodology & Pipeline Overview

1. **Candidate Building Selection (`building_selection_tool.ipynb`)**
   - Filters buildings by meter type (e.g., electricity / meter 0) and evaluates data completeness, missing timestamps, and continuous zero-reading streaks.
   - Highlights reliable target buildings such as Building 105 (full year of non-zero observations) and Building 1074.

2. **Data Extraction & Standardization (`building_data_extraction.ipynb`)**
   - Extracts targeted building and meter subsets from `train.csv` into dedicated subdirectories under `data/building/<building_id>/`.

3. **Exploratory Data Analysis & Weather Merging (`task1.ipynb`, `six_months.ipynb`)**
   - Analyzes diurnal, weekly, and seasonal consumption cycles via autocorrelation, boxplots, and heatmaps.
   - Merges hourly site weather observations (temperature, dew point, humidity) with building meter readings and imputes missing weather values.

4. **Feature Engineering**
   - **Temporal Features**: Hour of day, day of week, month, weekend indicators.
   - **Lag Features**: 24-hour lag (previous day, same hour) and 168-hour lag (previous week, same day and hour).
   - **Weather Features**: Ambient air temperature and rolling averages.

5. **Predictive Modeling & Evaluation**
   - **Model**: XGBoost Regressor trained on chronological splits (e.g., 80/20 train/test split or 6-month train windows) to avoid lookahead bias.
   - **Metrics**: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
   - **Baseline Comparison**: Benchmarked against a naive previous-week persistence baseline.

---

## Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies in a Python 3.10+ virtual environment:

```bash
# Clone the repository
git clone https://github.com/AryanBhanot/ashrae-energy-forecasting.git
cd ashrae-energy-forecasting

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Exploring Analysis Notebooks

Launch Jupyter Lab or Notebook to run the workflow:

```bash
jupyter lab
```

- **`notebooks/building_selection_tool.ipynb`**: Interactive building screening tool.
- **`notebooks/building_data_extraction.ipynb`**: Extract single-building slices from `train.csv`.
- **`notebooks/task1.ipynb`**: Complete end-to-end EDA, feature engineering, and modeling pipeline for Building 105.
- **`notebooks/six_months.ipynb`**: 6-month forecasting study and model serialization for Building 1074.

---

## Running Applications

### Interactive Streamlit Dashboard (Default App)

Launch the Streamlit app locally:

```bash
streamlit run apps/streamlit/app.py
```

The application will be accessible at `http://localhost:8501`.

### Flask Web Service

To run the Flask application:

```bash
flask --app apps/flask/app run
```

- Visit `http://127.0.0.1:5000/` for the home route.
- Visit `http://127.0.0.1:5000/hello/<name>` for the personalized dynamic template.

Alternatively, launch the debugger directly in VS Code using the pre-configured targets in `.vscode/launch.json`.

---

## Containerization & Cloud Deployment

The repository includes a lightweight `Dockerfile` configured for containerized execution and deployment to **Google Cloud Run** or any OCI-compliant container platform.

### Run Locally with Docker

```bash
# Build the container image
docker build -t ashrae-energy-forecasting .

# Run the container (maps port 8080 to container port 8080)
docker run -p 8080:8080 -e PORT=8080 ashrae-energy-forecasting
```

Access the application at `http://localhost:8080`.

### Deploy to Google Cloud Run

```bash
# Build and submit image to Google Artifact Registry / Container Registry
gcloud builds submit --tag gcr.io/<PROJECT_ID>/ashrae-energy-forecasting

# Deploy to Cloud Run
gcloud run deploy ashrae-energy-forecasting \
  --image gcr.io/<PROJECT_ID>/ashrae-energy-forecasting \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

---

## Reports & Deliverables

- **[Task1 Writeup](reports/Task1_Writeup.pdf)**: Detailed technical report documenting exploratory analysis findings, model architectures, baseline evaluations, and winter-break error analysis.
- **[Internship Progress Timeline](reports/internship_progress_timeline.md)**: Chronological summary of work packages and milestones completed.
