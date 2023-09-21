#!/usr/bin/env python
import argparse
import logging
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    logger.info(f"Starting uploading {args.file} ...")

    with wandb.init(
        project="census-classification",
        notes="training census dataset",
        tags=["training"],
        job_type="download_data",
    ) as run:
        with open(args.file, "r") as fp:

            logger.info("Creating artifact")
            artifact = wandb.Artifact(
                name=args.artifact_name,
                type=args.artifact_type,
                description=args.artifact_description,
                metadata={"original_file": args.file},
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

    go(args)
