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

        # if "basic_cleaning" in active_steps:
        #     _ = mlflow.run(
        #         os.path.join(hydra.utils.get_original_cwd(),
        #                      "src", "basic_cleaning"),
        #         "main",
        #         parameters={
        #             "input_artifact": config['basic_cleaning']['input_artifact'],
        #             "output_artifact": config['basic_cleaning']['output_artifact'],
        #             "output_type": config['basic_cleaning']['output_type'],
        #             "output_description": config['basic_cleaning']['output_description'],
        #             "min_price": config['etl']['min_price'],
        #             "max_price": config['etl']['max_price']
        #         },
        #     )

        # if "data_check" in active_steps:
        #     _ = mlflow.run(
        #         os.path.join(hydra.utils.get_original_cwd(),
        #                      "src", "data_check"),
        #         "main",
        #         parameters={
        #             "csv": config['data_check']['csv'],
        #             "ref": config['data_check']['ref'],
        #             "kl_threshold": config['data_check']['kl_threshold'],
        #             "min_price": config['etl']['min_price'],
        #             "max_price": config['etl']['max_price']
        #         },
        #     )

        # if "data_split" in active_steps:
        #     _ = mlflow.run(
        #         f"{config['main']['components_repository']}/train_val_test_split",
        #         "main",
        #         version='main',
        #         parameters={
        #             "input": config['data_check']['csv'],
        #             "test_size": config['modeling']['test_size'],
        #             "random_seed": config['modeling']['random_seed'],
        #             "stratify_by": config['modeling']['stratify_by']
        #         },
        #     )

        # if "train_random_forest" in active_steps:

        #     rf_config = os.path.abspath("rf_config.json")
        #     with open(rf_config, "w+") as config_file:
        #         json.dump(
        #             dict(config["modeling"]["random_forest"].items()), config_file)

        #     logger.info(f"modeling: {config['modeling']}")
        #     _ = mlflow.run(
        #         os.path.join(hydra.utils.get_original_cwd(),
        #                      "src", "train_random_forest"),
        #         "main",
        #         parameters={
        #             "trainval_artifact": config['modeling']['input_artifact'],
        #             "val_size": config['modeling']['val_size'],
        #             "random_seed": config['modeling']['random_seed'],
        #             "stratify_by": config['modeling']['stratify_by'],
        #             "rf_config": rf_config,
        #             "max_tfidf_features": config['modeling']['max_tfidf_features'],
        #             "output_artifact": config['modeling']['output_artifact']
        #         },
        #     )

        # if "test_regression_model" in active_steps:

        #     _ = mlflow.run(
        #         f"{config['main']['components_repository']}/test_regression_model",
        #         "main",
        #         version='main',
        #         parameters={
        #             "mlflow_model": "random_forest_export:prod",
        #             "test_dataset": "test_data.csv:latest"
        #         },
        #     )


if __name__ == "__main__":
    run_pipeline()
