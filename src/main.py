import json
import logging
import tempfile
import os
import mlflow
import wandb
import hydra
from omegaconf import DictConfig
import utils

_steps = [
    "download",
    "upload",
    "train_test_split",
    "data_check",
    "model_check",
    "model_training",
    "model_evaluation",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_name="config", config_path="conf")
def run_pipeline(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory():

        if "download" in active_steps:
            # Download file and load in W&B
            download_path = os.path.join(
                hydra.utils.get_original_cwd(),
                config["main"]["components_repository"],
                "data_preparation",
            )

            _ = mlflow.run(
                download_path,
                "main",
                parameters={"local_directory": "data"},
            )

        if "upload" in active_steps:
            # Upload file in W&B
            upload_path = os.path.join(
                hydra.utils.get_original_cwd(),
                config["main"]["components_repository"],
                "data_upload",
            )

            file_path = os.path.join(
                hydra.utils.get_original_cwd(), "../data/census.csv"
            )
            _ = mlflow.run(
                upload_path,
                "main",
                parameters={
                    "file": file_path,
                    "artifact_name": "census.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )

        if "train_test_split" in active_steps:
            file_path = os.path.join(hydra.utils.get_original_cwd(), "..")
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    config["main"]["components_repository"],
                    "data_split",
                ),
                "main",
                parameters={
                    "file_path": file_path,
                    "input_artifact": config["data_split"]["input_artifact"],
                    "artifact_root": config["data_split"]["artifact_root"],
                    "artifact_type": config["data_split"]["artifact_type"],
                    "test_size": config["data_split"]["test_size"],
                    "random_state": config["data_split"]["random_state"],
                    "stratify": config["data_split"]["stratify"],
                },
            )

        if "data_check" in active_steps:
            file_path = os.path.join(hydra.utils.get_original_cwd(), "test", "data")
            logger.info(f"data_check file_path: {file_path}")
            _ = mlflow.run(
                file_path,
                "main",
                parameters={
                    "input_artifact": config["data_check"]["input_artifact_full"],
                },
            )

            _ = mlflow.run(
                file_path,
                "main",
                parameters={
                    "input_artifact": config["data_check"]["input_artifact_train"],
                },
            )

            _ = mlflow.run(
                file_path,
                "main",
                parameters={
                    "input_artifact": config["data_check"]["input_artifact_test"],
                },
            )

        if "model_check" in active_steps:
            file_path = os.path.join(hydra.utils.get_original_cwd(), "test", "model")
            logger.info(f"model check file_path: {file_path}")
            _ = mlflow.run(
                file_path,
                "main",
                parameters={"input_artifact": config["model_check"]["input_artifact"]},
            )

        if "model_training" in active_steps:
            logger.info(f"model training: {config['model_training']['model_name']}")
            file_path = os.path.join(
                hydra.utils.get_original_cwd(), "model_build", "model_training"
            )
            logger.info(f"model training file_path: {file_path}")
            model_dir = os.path.join(hydra.utils.get_original_cwd(), "../model")
            logger.info(f"model training model_dir: {model_dir}")
            _ = mlflow.run(
                file_path,
                "main",
                parameters={
                    "model_directory": model_dir,
                    "artifact_root": config["model_training"]["artifact_root"],
                    "project_name": config["main"]["project_name"],
                    "model_file": config["model_training"]["model_file"],
                },
            )

        if "model_evaluation" in active_steps:
            file_path = os.path.join(
                hydra.utils.get_original_cwd(), "model_build", "model_evaluate"
            )
            logger.info(f"model evaluation file_path: {file_path}")
            model_dir = os.path.join(hydra.utils.get_original_cwd(), "../model")
            logger.info(f"model training model_dir: {model_dir}")
            _ = mlflow.run(
                file_path,
                "main",
                parameters={
                    "model_directory": model_dir,
                    "artifact_root": config["model_evaluation"]["artifact_root"],
                    "project_name": config["main"]["project_name"],
                    "model_file": config["model_evaluation"]["model_file"],
                },
            )


if __name__ == "__main__":
    run_pipeline()
