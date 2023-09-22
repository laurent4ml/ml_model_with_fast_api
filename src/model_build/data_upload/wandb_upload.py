#!/usr/bin/env python
import argparse
import logging
import os
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def upload(args):
    """upload creates a wandb.Artifact and adds a file to it

    Args:
        - file (str): file to upload to wandb
        - artifact_name (str): name of the artifact
        - artifact_type (str): type of artifact
        - artifact_description (str): description of the artifact
    """
    file_path = args.file
    logger.info(f"Starting uploading {file_path} ...")

    if not os.path.exists(file_path):
        logger.info(f"file {file_path} does not exits")
        exit(1)

    with wandb.init(
        project="census-classification",
        notes="training census dataset",
        tags=["training"],
        job_type="upload_data",
    ) as run:
        with open(file_path, "r") as fp:

            logger.info("Creating artifact")
            artifact = wandb.Artifact(
                name=args.artifact_name,
                type=args.artifact_type,
                description=args.artifact_description,
                metadata={"original_file": file_path},
            )
            artifact.add_file(fp.name, name="census_clean")

            logger.info("Logging artifact")
            run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file and upload it as an artifact to W&B",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--file", type=str, help="full path to the input file", required=True
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    upload(args)
