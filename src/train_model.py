# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import wandb
from sklearn.preprocessing import LabelBinarizer

# @TODO initialize wand, example:  run = wandb.init(job_type=\"download_data\", save_code=True)

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
X_train, X_val = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    X_train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
labelBinarizer = LabelBinarizer()
X_val, y_val, encoder, lb = process_data(
    X_val, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=labelBinarizer
)
# Train and save a model.
model = train_model(X_train, y_train)

# @TODO save model im wandb
