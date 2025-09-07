# Credit Card Approval Predictor

This project builds a machine learning model to predict whether a credit card application will be approved.

## Dataset

The project uses the [Credit Card Approval dataset](http://archive.ics.uci.edu/ml/datasets/credit+approval) from the UCI Machine Learning Repository. This dataset contains anonymized data about credit card applications.

## Project Overview

The project is implemented in a Jupyter Notebook (`notebook.ipynb`). The notebook walks through the following steps:

1.  **Data Loading and Inspection:** The dataset is loaded using pandas and inspected to understand its structure and identify any issues.
2.  **Data Preprocessing:**
    *   Missing values (marked as '?') are handled. Numeric missing values are imputed with the mean, and categorical missing values are imputed with the most frequent value.
    *   Non-numeric features are converted to a numeric format using one-hot encoding.
    *   Feature values are scaled to a uniform range (0-1) using `MinMaxScaler`.
3.  **Model Training:**
    *   The dataset is split into training and testing sets.
    *   A logistic regression model is trained on the preprocessed training data.
4.  **Model Evaluation:**
    *   The model's performance is evaluated on the test set using accuracy and a confusion matrix.
5.  **Hyperparameter Tuning:**
    *   `GridSearchCV` is used to find the best hyperparameters for the logistic regression model to improve its performance.

## How to Run

1.  Ensure you have the required libraries installed, including:
    *   pandas
    *   numpy
    *   scikit-learn
    *   jupyter

    You can install them using pip:
    ```bash
    pip install pandas numpy scikit-learn jupyter
    ```

2.  Clone this repository.
3.  Launch Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4.  Open `notebook.ipynb` and run the cells.
