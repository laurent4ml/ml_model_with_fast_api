import numpy as np
import logging
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def process_data(
    X,
    categorical_features=[],
    label=None,
    training=True,
    encoder=None,
    label_binarizer=None,
):
    """Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    label_binarizer : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    label_binarizer : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """
    logging.info("process_data - start")
    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    x_categorical = X[categorical_features].values
    logging.info(f"process_data - x_categorical.shape: {x_categorical.shape}")
    logging.info(f"process_data - x_categorical: {x_categorical}")
    x_continuous = X.drop(*[categorical_features], axis=1)
    logging.info(f"process_data - x_continuous.shape: {x_continuous.shape}")
    logging.info(f"process_data - x_continuous: {x_continuous}")
    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        label_binarizer = LabelBinarizer()
        x_categorical = encoder.fit_transform(x_categorical)
        logging.info(f"process_data - encoder_categories: {encoder.categories_}")
        y = label_binarizer.fit_transform(y.values).ravel()
    else:
        # encoder.transform
        # Input: Xarray-like of shape (n_samples, n_features)
        # Output: X_out{ndarray, sparse matrix} of shape (n_samples,
        # n_encoded_features)
        x_categorical = encoder.transform(x_categorical)
        try:
            y = label_binarizer.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([x_continuous, x_categorical], axis=1)
    logging.info(f"process_data - X.shape: {X.shape}")
    return X, y, encoder, label_binarizer
