# Titanic Survival Prediction

This repository contains code and notebooks for exploring the Kaggle Titanic dataset and building baseline models to predict passenger survival.

## Data Overview

Raw CSV files are stored in `data/raw`. The main training file is `train.csv` which includes passenger features such as age, sex, ticket class and survival label.

## Notebooks

- `01_data_exploration.ipynb` – initial look at the data.
- `02_eda.ipynb` – exploratory data analysis covering survival rates by feature, missing data patterns, distributions and correlations.
- `03_baseline_models.ipynb` – logistic regression, decision tree and random forest evaluated with cross-validation.

## Model Pipeline

Feature engineering and preprocessing are implemented in `src/data_preprocessing.py`. The module handles missing values, derives additional features like `FamilySize` and `Title`, encodes categorical variables and scales numeric attributes. It exposes helper functions to fit and apply the preprocessing pipeline.

An end-to-end training and prediction workflow is available in
`src/automated_pipeline.py`.  The pipeline reads configuration from
`config/pipeline_config.json`, trains a selection of models, logs
performance metrics and writes a Kaggle submission file to the
`submissions/` directory.

To execute the full automated pipeline run:

```bash
python -m src.automated_pipeline
```

Performance metrics for each run are appended to
`results/performance_log.csv` and can be analysed with the utilities in
`src/performance_monitor.py`.

For experimenting with different submission strategies and probability
calibration techniques refer to `src/submission_optimizer.py`.

## Running the Code

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the notebooks with Jupyter to reproduce the analysis and model results:
   ```bash
   jupyter notebook notebooks/02_eda.ipynb
   ```

### Testing

Unit tests cover key utility functions, data loading and the automated
pipeline.  Run the full test suite with:

```bash
pytest
```

## Results and Insights

- Female passengers and those in higher classes show markedly higher survival rates.
- Baseline models reach reasonable accuracy using cross-validation; see the baseline notebook for detailed metrics.

## License

MIT
