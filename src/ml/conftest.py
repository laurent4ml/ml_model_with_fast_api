import pytest
import pandas as pd
import wandb

# from src.ml.data import process_data

run = wandb.init(project="census-classification", job_type="data_tests")


def pytest_addoption(parser):
    parser.addoption("--input_artifact", action="store")


@pytest.fixture(scope="session")
def training_data(request):
    input_artifact = request.config.option.input_artifact
    if input_artifact is None:
        pytest.fail("--input_artifact missing on command line")
    local_path = run.use_artifact(input_artifact).file()
    return pd.read_csv(local_path)


"""
@pytest.fixture(scope="session")
def training_data(artifact_data):
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

    x_train, y_train = process_data(
        training_data, categorical_features=cat_features, label="income", training=True
    )
    return x_train, y_train
"""
