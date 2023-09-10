import wandb
import pandas as pd

# This is global so all tests are collected under the same
# run
run = wandb.init(project="census-classification", job_type="data_tests")


def test_column_presence_and_type(data):

    # A dictionary with the column names as key and a function that verifies
    # the expected dtype for that column. We do not check strict dtypes (like
    # np.int32 vs np.int64) but general dtypes (like is_integer_dtype)
    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_string_dtype,
        "fnlwgt": pd.api.types.is_integer_dtype,
        "education": pd.api.types.is_string_dtype,
        "education-num": pd.api.types.is_integer_dtype,
        "marital-status": pd.api.types.is_string_dtype,
        "occupation": pd.api.types.is_string_dtype,
        "relationship": pd.api.types.is_string_dtype,
        "race": pd.api.types.is_string_dtype,
        "sex": pd.api.types.is_string_dtype,
        "capital-gain": pd.api.types.is_integer_dtype,
        "capital-loss": pd.api.types.is_integer_dtype,  # This is integer, not float as one might expect
        "hours-per-week": pd.api.types.is_integer_dtype,
        "native-country": pd.api.types.is_string_dtype,
    }

    # Check column presence
    assert set(data.columns.values).issuperset(set(required_columns.keys()))

    # Check that the columns are of the right dtype
    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(
            data[col_name]
        ), f"Column {col_name} failed test {format_verification_funct}"


def test_class_names(data):

    known_classes = [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ]

    assert (
        data["relationship"].isin(known_classes).all
    ), "Column relationship failed test class names"


def test_column_ranges(data):

    ranges = {
        "age": (1, 120),
        "fnlwgt": (0, 10000000),
        "education-num": (0, 1000000),
        "capital-gain": (0, 9000000),
        "capital-loss": (0, 5000000),
        "hours-per-week": (0, 150),
    }

    for col_name, (minimum, maximum) in ranges.items():
        assert data[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed test column ranges",
            f"Should be between {minimum} and {maximum}",
            f"Is actually bettween {data[col_name].min()} and {data[col_name].max()}",
        )
