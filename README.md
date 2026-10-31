# Customer Churn Prediction — Model 1

## Overview
This repository contains the first machine-learning model for the project: a customer churn prediction model built with **LightGBM**.

The model estimates whether a customer is at risk of churn based on historical purchase and activity behavior.

## Model
- **Algorithm:** LightGBM binary classifier (`LGBMClassifier`)
- **Target:** `churn` (1 = churn, 0 = no churn)
- **Decision threshold stored with the model:** `0.30`
- **Saved artifact:** `my_churn_model.pkl`

The saved artifact is a Joblib/Pickle bundle containing:
- `model`: the trained LightGBM model
- `features`: the ordered feature names expected by the model
- `threshold`: the decision threshold
- `model_type`: model type (`LightGBM`)

## Notebook
`churn_model.ipynb` contains the data preparation, feature engineering, model training, evaluation, exploratory analysis, and model-export workflow.

The notebook uses purchase and customer-activity events to construct features such as:
- Purchase frequency and recency
- Add-to-cart and remove-from-cart activity
- Page visits
- Search activity
- Recent activity windows (including 7-day and 30-day counts)

It evaluates the model using classification metrics, including ROC-AUC, PR-AUC, precision, recall, F1-score, and a confusion matrix.

## Required data
The notebook expects the source data to be available in Google Drive under:

`/content/drive/MyDrive/Synerise/churn_experiment`

The expected source files include:

### `input/`
- `product_buy.parquet`
- `add_to_cart.parquet`
- `remove_from_cart.parquet`
- `page_visit.parquet`
- `search_query.parquet`

### `target/`
- `train_target.parquet`
- `validation_target.parquet`

The notebook also creates intermediate Parquet files under the `purchase_prediction` directory.

**The source datasets are not included in this repository.** They must be obtained through the project's approved data-sharing process. Do not upload customer-level or confidential data to this public repository.

## Environment and dependencies
The notebook was developed for **Google Colab** and uses its `google.colab` integration to mount Google Drive and download the saved model.

Install the Python packages listed in [`requirements.txt`](requirements.txt). Google Colab provides its own `google.colab` module; it is not a regular PyPI dependency for a local Python environment.

## Running the notebook
1. Open `churn_model.ipynb` in Google Colab.
2. Make sure you have authorized access to the required project data.
3. Confirm that the data files are available under the directory described in **Required data**.
4. Run the notebook cells in order to build the features, train/evaluate the model, and save the model bundle.
5. The final export cell saves `my_churn_model.pkl`.

The notebook currently uses a Google Drive path specific to its original environment. If your Drive folder differs, update the `BASE` path near the beginning of the notebook.

## Using the saved model
The saved file is a serialized Python object. Load it only from a trusted source and with compatible versions of the required libraries.

Example:

```python
import joblib
import pandas as pd

bundle = joblib.load("my_churn_model.pkl")

model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

# `new_data` must contain every required feature column.
X = new_data[features].astype(float)

churn_probabilities = model.predict_proba(X)[:, 1]
churn_predictions = (churn_probabilities >= threshold).astype(int)
```

`new_data` must be prepared using the same feature definitions and preprocessing used during training. The saved model file alone does not generate the required features from raw event files.

## Repository contents
- `churn_model.ipynb` — model development and training notebook
- `my_churn_model.pkl` — saved trained model bundle
- `README.md` — project and usage documentation
- `requirements.txt` — Python package dependencies

## Notes
- This is the first model component. Additional project components may be added to the repository as they become available.
- Do not commit credentials, API keys, private customer data, or confidential source datasets.
- Clear notebook outputs that may contain customer identifiers or customer-level records before committing a cleaned notebook.
