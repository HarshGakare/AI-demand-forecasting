# AI-Powered Demand Forecasting and Inventory Optimization System

An end-to-end data science project that forecasts retail demand and translates predictions into inventory planning insights using the Rossmann Store Sales dataset.

## 1. Project Overview

This project combines data engineering, machine learning, time-series forecasting, inventory optimization, data validation, model explainability, and application development to support retail demand planning.

The system processes historical retail sales data, engineers forecasting features, evaluates predictive models, and converts demand forecasts into inventory planning insights.

### Key Features

- Data loading, merging, cleaning, and validation
- Feature engineering for retail demand forecasting
- LightGBM and Random Forest model comparison
- Time-series rolling cross-validation
- Multi-horizon and recursive forecasting experiments
- Forecast evaluation using MAE, RMSE, and WAPE
- Inventory optimization and business KPI reporting
- SHAP-based model explainability
- FastAPI application for forecasting services
- Streamlit dashboard for interactive analysis
- MySQL database integration
- Automated testing and report-integrity verification

## 2. Technology Stack

- **Language:** Python
- **Data processing:** pandas, NumPy
- **Machine learning:** scikit-learn, LightGBM
- **Forecasting:** Time-series validation, multi-horizon forecasting, recursive forecasting
- **Explainability:** SHAP
- **Database:** MySQL
- **API:** FastAPI
- **Dashboard:** Streamlit
- **Testing:** pytest

## 3. Dataset

This project uses the Rossmann Store Sales dataset.

Place the following files in `data/raw/`:

- `train.csv`
- `test.csv`
- `store.csv`
- `sample_submission.csv`

The dataset is available through the Rossmann Store Sales Kaggle competition. Obtain the files from the original competition and follow its applicable terms.

## 4. Project Structure

```text
AI-demand-forecasting/
├── api/
├── config/
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── sql/
├── src/
│   ├── data/
│   ├── forecasting/
│   ├── inventory/
│   ├── models/
│   └── nlp/
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

The `src/` directory contains the core data science pipeline, data validation, feature engineering, forecasting, modeling, inventory optimization, and reporting modules. The `api/`, `dashboard/`, `sql/`, `config/`, and `tests/` directories support the application, database, configuration, and testing workflows.

## 5. Environment Setup

Run the commands in this section from the project root using PowerShell.

### Create and activate a virtual environment

If you already have a working `.venv`, you can skip this step.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Configure environment variables

Configure the `.env` file and any required application settings for your local environment, particularly database credentials and connection settings.

**Security:** Never commit database passwords, API keys, tokens, or other secrets to version control.

## 6. Run the Data Pipeline

Run the main pipeline from the project root:

```powershell
python -m src.pipeline
```

The pipeline runs the configured data processing and forecasting workflow, including data preparation, feature engineering, validation, and associated project steps.

Processed datasets are stored under `data/processed/`, while generated evaluation results and reports are stored under `reports/`.

## 7. Run the Automated Tests

Run the test suite:

```powershell
python -m pytest -v
```

### Latest recorded test result

- **Tests passed:** 70
- **Tests failed:** 0
- **Warnings:** 2

The two recorded warnings concern a Starlette test-client dependency deprecation and a pandas dtype-selection compatibility warning. These should be reviewed as dependencies evolve.

## 8. Verify Project Outputs

### Verify feature outputs

```powershell
python -m src.verify_feature_outputs
```

### Verify report integrity

```powershell
python -m src.verify_project_reports
```

The latest recorded report-integrity check verified 12 reports and found no issues.

Generated reports include model comparison results, cross-validation summaries, experiment tracking, model evaluation metrics, recursive forecasting metrics, inventory summaries, data quality results, and SHAP feature importance.

## 9. Model Evaluation

The project uses the following evaluation metrics:

- **MAE — Mean Absolute Error:** Measures the average absolute difference between actual and predicted sales.
- **RMSE — Root Mean Squared Error:** Penalizes larger prediction errors more heavily.
- **WAPE — Weighted Absolute Percentage Error:** Measures total absolute error relative to total actual demand.

### Recorded model results

| Evaluation | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| LightGBM holdout | 580.42 | 854.78 | 9.55% |
| LightGBM 30-day recursive forecast | 1,989 | 2,819 | 32.72% |

### Same-fold rolling cross-validation

| Model | Mean MAE | Mean RMSE | Mean WAPE |
|---|---:|---:|---:|
| LightGBM | 413.11 | 601.81 | 7.09% |
| Random Forest | 418.56 | 659.03 | 7.15% |

In the recorded same-fold rolling cross-validation experiments, LightGBM achieved slightly better average error metrics, while Random Forest showed lower variability.

These results are historical experiment outputs, not guarantees of future performance. Metrics from different evaluation setups should not be compared as if they were measured on the same data and forecasting horizon.

## 10. Inventory Optimization

The project includes inventory optimization logic intended to translate demand forecasts into inventory planning recommendations and business KPIs.

Inventory recommendations depend on the forecasting horizon, demand assumptions, and configured inventory parameters.

**Important limitation:** Inventory quantities may use proxy stock assumptions rather than verified real-world stock-on-hand data. Recommendations should therefore be treated as analytical estimates until connected to reliable inventory, lead-time, replenishment, and service-level data.

## 11. Data Quality and Imputation

The project includes data validation and reporting to identify missing values and other data quality issues.

### Missing `Open` values

Eleven missing `Open` values in the Rossmann test data for Store 622 were inferred as `1` based on historical weekday opening patterns.

This is a documented inference, not verified ground truth. The affected rows and imputation method are recorded in the project audit output.

### Missing competition and promotion metadata

Some competition and Promo2 metadata fields contain missing values. These can indicate that a competition or promotion was not applicable or that information was not recorded.

Such values should be handled according to the meaning of each feature rather than blindly replacing every missing value with a single default.

## 12. API and Dashboard

The project includes:

- **FastAPI:** API endpoints for the configured forecasting workflow.
- **Streamlit:** An interactive dashboard for project analysis and forecasting-related outputs.
- **MySQL:** Database integration for the configured data workflow.

The exact launch commands and required environment variables depend on the current application entry points and configuration. Check the relevant files in `api/`, `dashboard/`, and `src/` before starting the services.

## 13. Limitations

- Forecast accuracy can deteriorate over longer recursive forecasting horizons.
- The recorded Store 622 `Open` values are inferred rather than verified.
- Inventory optimization outputs may rely on proxy stock assumptions.
- Forecast performance depends on data quality, evaluation windows, feature availability, and model configuration.
- The recorded experimental metrics may change when the pipeline, data, dependencies, or model parameters change.
- Database and application functionality depends on correct local configuration.

## 14. Reproducibility

For repeatable experiments:

1. Use the configured Python environment.
2. Install dependencies from `requirements.txt`.
3. Keep the required raw dataset files in `data/raw/`.
4. Configure required environment variables.
5. Run the pipeline and automated tests.
6. Review generated reports before comparing results between runs.

## 15. License and Attribution

Review the Rossmann dataset's applicable Kaggle competition terms before redistributing the dataset or derived materials.

Add an appropriate software license to this repository if you intend to distribute the project. Do not imply that the project has a license until one has been selected and added.
