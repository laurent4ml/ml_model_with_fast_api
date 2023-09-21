#!/usr/bin/env python
"""
Downloads data from given urls. Performs basic cleaning on the data
and save the results in a local directory.
"""
import argparse
import os
import requests
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def prepare_data(args):
    """
    Downloads data from given urls. Performs basic cleaning on the data.
    - replacing "?" with np.nan
    - removing all "na"
    Save the results in a local directory

    ARGS:
        path: str - local directory to store files
    """
    logger.info(f"data preparation - local directory set as {args.local_directory}")

    urls = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    )

    root = get_project_root()
    logger.info(f"root: {root}")
    artifact_local_directory = os.path.join(root, args.local_directory)
    if not os.path.exists(artifact_local_directory):
        os.mkdir(artifact_local_directory)

    # Download input artifact
    logger.info("Downloading input artifact")
    for url in urls:
        response = requests.get(url)
        name = os.path.basename(url)
        logger.info(f"Processing {name}")
        with open(os.path.join(artifact_local_directory, name), "wb") as f:
            f.write(response.content)

    names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    data = pd.read_csv(
        artifact_local_directory + "/adult.data", names=names, skipinitialspace=True
    )

    logger.info("data.replace('?', np.nan)")
    for col in names:
        data[col] = data[col].apply(lambda x: np.nan if x == "?" else x)

    data.dropna(inplace=True)

    # Save dataset to local csv file
    logger.info("save dataset to local csv file")
    output_folder = os.path.join(artifact_local_directory, "census.csv")
    data.to_csv(output_folder, sep=",", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")

    parser.add_argument(
        "--local_directory",
        type=str,
        help="please insert a local directory",
        required=True,
    )

    args = parser.parse_args()

    prepare_data(args)
