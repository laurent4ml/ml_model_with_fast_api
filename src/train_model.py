# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import logging
import pandas as pd
import wandb
import joblib
import argparse
import os
from ml.data import process_data
from ml.model import train_model
from sklearn.preprocessing import LabelBinarizer

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(project=args.project_name, job_type="process_data")

    logger.info("Downloading artifact from  WandB")
    artifact = run.use_artifact(args.artifact_training)
    artifact_path = artifact.file()

    logger.info("Uploading training data") 
    training_data = pd.read_csv(artifact_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    #logger.info("Splitting the data into train and test")
    #X_train, X_val = train_test_split(data, test_size=0.20)

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

    logger.info("Processing data")
    X_train, y_train, encoder, lb = process_data(
        training_data, categorical_features=cat_features, label="income", training=True
    )

    # Process the test data with the process_data function.
    #labelBinarizer = LabelBinarizer()
    #X_val, y_val, encoder, lb = process_data(
    #    X_val, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=labelBinarizer
    #)

    # Train and save a model.
    logger.info("Training model")
    model = train_model(X_train, y_train)

    model_path = os.path.join("model", args.model_file)

    # save model
    logger.info("Saving Model")
    joblib.dump(model, model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--artifact_training",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--project_name", type=str, help="Name for the project", required=True
    )

    parser.add_argument(
        "--model_file", type=str, help="Model file name", required=True
    )

    args = parser.parse_args()

    go(args)
