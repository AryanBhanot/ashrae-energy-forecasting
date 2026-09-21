# ASHRAE Energy Forecasting

This project forecasts hourly building electricity consumption using ASHRAE meter and weather data, featuring exploratory data analysis, time-series feature engineering, XGBoost predictive modeling, and a Flask web service.

## Project Structure

```text
ashrae-energy-forecasting/
├── data/                                 # Datasets and metadata
│   ├── building/                         # Extracted building meter datasets (e.g., 105, 1074)
│   ├── building_metadata(in).csv         # Metadata for ASHRAE buildings
│   └── weather_train.csv                 # Hourly site weather observations
├── apps/                                 # User interfaces and web services
│   └── flask/                            # Flask web service
│       ├── static/                       # Static assets
│       │   └── site.css                  # Styling for templates
│       ├── templates/                    # Jinja2 HTML templates
│       │   └── hello_there.html          # Dynamic greeting template
│       └── app.py                        # Flask server entry point
├── models/                               # Serialized machine learning models
│   └── building_1074_six_months_model.pkl# Trained XGBoost pipeline
├── notebooks/                            # Jupyter notebooks for analysis and ML
│   ├── building_selection_tool.ipynb     # Interactive tool to filter & select reliable candidate buildings
│   ├── six_months.ipynb                  # 6-month forecasting analysis and modeling for Building 1074
│   └── task1.ipynb                       # Main pipeline: EDA, feature engineering, XGBoost & baseline evaluation
├── reports/                              # Documentation and deliverables
│   ├── Task1_Writeup.pdf                 # Final writeup report
│   └── internship_progress_timeline.md   # Chronological work summary
├── .gitignore                            # Git ignore rules for clean repository hygiene
├── requirements.txt                      # Project Python dependencies
└── README.md                             # Project overview and documentation
```

## Getting Started

### 1. Installation

Clone the repository and install the dependencies in a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Analysis Notebooks

Launch Jupyter Notebook or JupyterLab to interact with the notebooks:

```bash
jupyter lab
```

- `notebooks/task1.ipynb`: Full end-to-end workflow on Building 105 (data ingestion, weather merging, lag features, XGBoost model, evaluation against previous-week baseline).
- `notebooks/building_selection_tool.ipynb`: Interactive candidate building inspector.
- `notebooks/six_months.ipynb`: Modeling for Building 1074 with 6 months of training data.

### 3. Running the Flask App

To run the Flask application locally:

```bash
flask --app apps/flask/app run
```

Or open the project in VS Code and press `F5` to start using the pre-configured debugger in `.vscode/launch.json`.
Navigate to `http://127.0.0.1:5000/hello/<your-name>` to test the service.
