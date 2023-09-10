import logging
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

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
