import pytest
import pandas as pd
import wandb

run = wandb.init(project="census-classification", job_type="data_tests")


def pytest_addoption(parser):
    parser.addoption("--input_artifact", action="store")


@pytest.fixture(scope="session")
def data(request):
    input_artifact = request.config.option.input_artifact
    if input_artifact is None:
        pytest.fail("--input_artifact missing on command line")
    local_path = run.use_artifact(input_artifact).file()
    return pd.read_csv(local_path)
