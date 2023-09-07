import logging
import pandas as pd
import wandb
from joblib import dump
import argparse
import os
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def run_training_steps(args):
    """
    train - function to train and save a Logistic Regression model

    Args:
     - artifact_root: dataset root name
     - project_name: (str) name of the project
     - model_file: (str) model filename
    """
    run = wandb.init(project=args.project_name, job_type="process_data")

    logger.info("Downloading training artifact from  WandB")
    artifact_training_name = (
        f"laurent4ml/{args.project_name}/{args.artifact_root}_train.csv:latest"
    )
    logger.info(f"Artifact Training Name: {artifact_training_name}")
    artifact_training = run.use_artifact(artifact_training_name)
    artifact_training_path = artifact_training.file()

    logger.info("Uploading training data")
    training_data = pd.read_csv(artifact_training_path)

    logger.info("Downloading test artifact from WandB")
    artifact_test_name = (
        f"laurent4ml/{args.project_name}/{args.artifact_root}_test.csv:latest"
    )
    artifact_test = run.use_artifact(artifact_test_name)
    artifact_test_path = artifact_test.file()

    logger.info("Uploading test data")
    test_data = pd.read_csv(artifact_test_path)

    # categorical features
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

    logger.info("Processing training data")
    x_train, y_train, encoder, label_binarizer = process_data(
        training_data, categorical_features=cat_features, label="income", training=True
    )
    logger.info("x_train.shape:")
    logger.info(x_train.shape)
    logger.info(f"Type of x_train: {type(x_train)}")
    logger.info("Encoder Categories:")
    for enc_cat in encoder.categories_:
        logger.info(enc_cat)
    logger.info("Label Binarizer Classes:")
    logger.info(label_binarizer.classes_)

    # Process the test data with the process_data function.
    logger.info("Processing test data")
    x_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=cat_features,
        label="income",
        training=False,
        encoder=encoder,
        label_binarizer=label_binarizer,
    )

    # Train and save a model.
    logger.info("Training model")
    model = train_model(x_train, y_train)

    # save model
    logger.info("Saving Model")
    model_path = os.path.join("model", args.model_file)
    dump(model, model_path)

    # run predictions
    logger.info(f"Running predictions for model: {model_path}")
    preds = inference(model_path, x_test)

    ## evaluate model
    logger.info("Evaluating Model")
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fbeta: {fbeta}")
    run.log({"precision": precision, "recall": recall, "fbeta": fbeta})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model and store results in WandB",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="name for the root artifact",
        required=True,
    )

    parser.add_argument(
        "--project_name", type=str, help="Name for the project", required=True
    )

    parser.add_argument("--model_file", type=str, help="Model file name", required=True)

    arguments = parser.parse_args()

    run_training_steps(arguments)
