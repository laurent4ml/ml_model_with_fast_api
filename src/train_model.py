import logging
import pandas as pd
import wandb
from joblib import dump
import argparse
import os
import wandb
from ml.data import process_data
from ml.model import train_model

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
    logging.info(f"train_model - training_data.shape: {training_data.shape}")
    logging.info(f"train_model - training_data.columns: {training_data.columns}")

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

    # Train and save a model.
    logger.info("Training model")
    model = train_model(x_train, y_train)

    # save model
    logger.info("Saving Model")
    model_path = os.path.join("model", args.model_file)
    dump(model, model_path)

    # save encoder
    logger.info("Saving Encoder")
    encoder_path = os.path.join("model", "onehot_encoder.joblib")
    dump(encoder, encoder_path)

    # save label binarizer
    logger.info("Saving Label Binarizer")
    label_binarizer_path = os.path.join("model", "label_binarizer.joblib")
    dump(label_binarizer, label_binarizer_path)

    # store model in Weight and Biases
    model_artifact = wandb.Artifact(f"{args.project_name}-model_{run.id}", type="model")
    model_artifact.add_file(model_path)
    run.log_artifact(model_artifact)

    # Link the model to the Model Registry
    run.link_artifact(
        model_artifact, f"laurent4ml/model-registry/{args.project_name}-model"
    )

    # store encoder in Weight and Biases
    encoder_artifact = wandb.Artifact(
        f"{args.project_name}-encoder_{run.id}", type="model"
    )
    encoder_artifact.add_file(encoder_path)
    run.log_artifact(encoder_artifact)

    # Link the model to the Model Registry
    run.link_artifact(
        encoder_artifact, f"laurent4ml/model-registry/{args.project_name}-encoder"
    )

    run.finish()


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
