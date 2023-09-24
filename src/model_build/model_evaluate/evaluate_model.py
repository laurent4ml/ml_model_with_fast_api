import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import wandb
import argparse
import os
import sys

# append the path of the parent directory
sys.path.append("../../..")
from src.ml.data import process_data
from src.ml.model import compute_model_metrics, inference

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def get_slice_stats_per_feature(df, feature):
    """Function for calculating descriptive stats on slices of one feature

    Args:
        df (np.ndarray): dataset with classes
        feature (str): feature to study

    Returns:
        slice_metrics: performance metrics for slices

    Function for calculating descriptive stats on slices of the Iris dataset.
    """
    slice_metrics = {}
    logger.info(f"Feature: {feature}")
    logger.info(f"df shape: {df.shape}")
    for cls in df["sex"].unique():
        logger.info(f"gender slice: {cls}")
        slice_metrics[cls] = {}
        df_temp = df[df["sex"] == cls]
        logger.info(f"df_temp shape: {df_temp.shape}")
        mean = df_temp[feature].mean()
        logger.info(f"mean: {mean}")
        stddev = df_temp[feature].std()
        logger.info(f"std dev: {stddev}")
        # store mean and standard deviation
        slice_metrics[cls]["mean"] = mean
        slice_metrics[cls]["stddev"] = stddev
    return slice_metrics


def get_slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_dataset: pd.DataFrame,
    categorical_features: [str],
):
    """Get performance metrics for slices.

    Args:
        y_true (np.ndarray): ground truth labels
        y_pred (np.ndarray): predicted labels
        feature_dataset (pd.DataFrame): Dataframe containing the features.
        categorical_features (list): list of categorical features
    Returns:
        Dict: performance metrics for slices
    """
    slice_metrics = {}
    dataset = pd.concat(
        [feature_dataset, pd.DataFrame(y_true, columns=["y_true"])], axis=1
    )
    dataset = pd.concat([dataset, pd.DataFrame(y_pred, columns=["y_preds"])], axis=1)
    logger.info(f"dataset shape: {dataset.shape}")

    for col in dataset.columns:
        logger.info(f"column: {col}")

    for feature in categorical_features:
        slice_metrics[feature] = {}
        logger.info(f"feature: {feature}")
        slices = dataset.loc[:, feature].unique()
        # recurring on all slices in feature column
        for slice_name in slices:
            logger.info(f"slice name: {slice_name}")
            mask = dataset[dataset[feature] == slice_name]
            metrics = precision_recall_fscore_support(
                mask["y_true"], mask["y_preds"], average="micro"
            )
            logger.info(metrics)
            slice_metrics[feature][slice_name] = {}
            slice_metrics[feature][slice_name]["precision"] = metrics[0]
            slice_metrics[feature][slice_name]["recall"] = metrics[1]
            slice_metrics[feature][slice_name]["f1"] = metrics[2]

    return slice_metrics


def walkmetrics(data, run):
    for feature, results in data.items():
        logger.info("feature: {0} - {1}".format(feature, results))
        if isinstance(results, dict):
            for slice, result in results.items():
                logger.info("{0} - {1} - {2}".format(feature, slice, result))
                for r in result:
                    key = f"data-slice-{feature}"
                    run.log({key: result})

        else:
            logger.info("{0} : {1}".format(feature, results))


def run_evaluate_steps(args):
    """
    function to evaluate a Logistic Regression model

    Args:
     - artifact_root: dataset root name
     - project_name: (str) name of the project
     - model_file: (str) model filename
    """
    run = wandb.init(project=args.project_name, job_type="evaluate_model")

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

    logger.info("Downloading training artifact from  WandB")
    artifact_training_name = (
        f"laurent4ml/{args.project_name}/{args.artifact_root}_train.csv:latest"
    )
    logger.info(f"Artifact Training Name: {artifact_training_name}")
    artifact_training = run.use_artifact(artifact_training_name)
    artifact_training_path = artifact_training.file()

    logger.info("Uploading training data")
    training_data = pd.read_csv(artifact_training_path)

    logger.info("Processing training data")
    _, _, encoder, label_binarizer = process_data(
        training_data, categorical_features=cat_features, label="income", training=True
    )

    logger.info("Downloading test artifact from  WandB")
    artifact_test_name = (
        f"laurent4ml/{args.project_name}/{args.artifact_root}_test.csv:latest"
    )

    logger.info(f"Artifact Test Name: {artifact_test_name}")
    artifact_test = run.use_artifact(artifact_test_name)
    artifact_test_path = artifact_test.file()

    logger.info("Uploading test data")
    test_data = pd.read_csv(artifact_test_path)

    # Process test data with the process_data function.
    logger.info("Processing test data")
    x_test, y_test, _, _ = process_data(
        test_data,
        categorical_features=cat_features,
        label="income",
        training=False,
        encoder=encoder,
        label_binarizer=label_binarizer,
    )

    # run predictions
    model_path = os.path.join(args.model_directory, args.model_file)
    logger.info(f"Loading Model and running predictions for model: {model_path}")
    preds = inference(model_path, x_test)

    ## evaluate overall model performance
    logger.info("Evaluating overall model performance")
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fbeta: {fbeta}")
    run.log({"precision": precision, "recall": recall, "fbeta": fbeta})

    ## log descriptive stats per slice for the education and age features
    stats_features = ("education-num", "age")
    for stats_feature in stats_features:
        logger.info(f"Log descriptive stats for {stats_feature} on slice of data")
        class_slices = get_slice_stats_per_feature(training_data, stats_feature)
        for slice in class_slices:
            mean_key = f"{stats_feature}-{slice}-mean"
            stddev_key = f"{stats_feature}-{slice}-stddev"
            stddev = class_slices[slice]["stddev"]
            mean = class_slices[slice]["mean"]
            run.log({mean_key: mean, stddev_key: stddev})
            logger.info(f"{mean_key}: {mean}")
            logger.info(f"{stddev_key}: {stddev}")

    ## performance metrics per slice for the categorical features
    logger.info("performance metrics per slice for the categorical features")
    slice_metrics = get_slice_metrics(y_test, preds, test_data, cat_features)
    walkmetrics(slice_metrics, run)

    ## calculate auc
    auc = roc_auc_score(y_test, preds)
    logger.info(f"auc: {auc}")
    run.log({"auc": auc})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model and store results in WandB",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_directory",
        type=str,
        help="directory to save model files",
        required=True,
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

    run_evaluate_steps(arguments)
