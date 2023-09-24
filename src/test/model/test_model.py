# Pytest module for testing linear regression model function
from sklearn.pipeline import Pipeline
import sklearn
import pytest
from src.ml.data import process_data
from src.ml.model import train_model


@pytest.fixture(scope="session")
def cat_features():
    """Categorical features"""
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return categorical_features


def test_model_return_pipeline(cat_features, data):
    """
    Tests the returned object of the modeling function
    """
    x_train, y_train, _, _ = process_data(
        data, categorical_features=cat_features, label="income", training=True
    )

    pipe = train_model(x_train, y_train)

    assert isinstance(pipe, Pipeline)
    # Check the length of the returned object
    assert len(pipe) == 2


def test_model_type_is_correct(cat_features, data):
    """
    Tests the training model type
    """
    x_train, y_train, _, _ = process_data(
        data, categorical_features=cat_features, label="income", training=True
    )
    # Get back the trained model
    pipe = train_model(x_train, y_train)
    trained_model = pipe[1]
    # Type check for the returned object
    assert isinstance(trained_model, sklearn.linear_model.LogisticRegression)


def test_error_msg(cat_features, data):
    """
    Tests error msg
    """
    x_train, y_train, _, _ = process_data(
        data, categorical_features=cat_features, label="income", training=True
    )

    # AssertionError check for wrong input type
    msg = train_model("X", y_train)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Features must be a Numpy array"

    msg = train_model(x_train, "y")
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Targets must be a Numpy array"
