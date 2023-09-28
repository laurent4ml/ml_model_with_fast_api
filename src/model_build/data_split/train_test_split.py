#!/usr/bin/env python
import argparse
import logging
import os
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def split_dataset(args):
    """
    Downloading full dataset from wandb and splits it into train and test datasets.
    Store the train and test datasets on local and upload them to wandb.

    Args:
        - file_path (str): file path
        - input_artifact (str): Input artifact string
        - artifact_root (str): Artifact root
        - artifact_type (str): Artifact type
        - test_size (int): Test size
        - random_state  (int): Random state
        - stratify  (int): Random strat
    """
    run = wandb.init(project="census-classification", job_type="split_data")

    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path, low_memory=False)

    # Split model_dev/test
    logger.info("Splitting data into train and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df[args.stratify] if args.stratify != "null" else None,
    )

    for split, df in splits.items():

        # Make the artifact name from the provided root plus the name of the
        # split
        artifact_name = f"{args.artifact_root}_{split}.csv"

        local_directory = os.path.join(args.file_path, "data_split")

        if not os.path.exists(local_directory):
            logger.info(f"Creating {local_directory}")
            os.mkdir(local_directory)

        path = os.path.join(local_directory, artifact_name)
        logger.info(f"Saving the {split} dataset to {path}")

        # Save to local filesystem
        df.to_csv(path, index=False)

        artifact = wandb.Artifact(
            name=artifact_name,
            type=args.artifact_type,
            description=f"{split} split of dataset {args.input_artifact}",
        )
        artifact.add_file(path)
        # Upload to W&B
        logger.info(f"Logging artifact {artifact_name}  in WandB")
        run.log_artifact(artifact)

        # This waits for the artifact to be uploaded to W&B. If we
        # do not add this, the temp directory might be removed before
        # W&B had a chance to upload the datasets, and the upload
        # might fail
        artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--file_path",
        type=str,
        help="File path",
        required=True,
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_root",
        type=str,
        help="Root for the names of the produced artifacts. The script will produce 2 artifacts: "
        "{root}_train.csv and {root}_test.csv",
        required=True,
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the produced artifacts",
        required=True,
    )

    parser.add_argument(
        "--test_size",
        help="Fraction of dataset or number of items to include in the test split",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--random_state",
        help="An integer number to use to init the random number generator. It ensures repeatibility in the"
        "splitting",
        type=int,
        required=False,
        default=42,
    )

    parser.add_argument(
        "--stratify",
        help="If set, it is the name of a column to use for stratified splitting",
        type=str,
        required=False,
        default="null",  # unfortunately mlflow does not support well optional params
    )

    args = parser.parse_args()

    split_dataset(args)
