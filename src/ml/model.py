import logging
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import load
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    try:
        assert isinstance(X_train, np.ndarray), "Features must be a Numpy array"
        assert isinstance(y_train, np.ndarray), "Targets must be a Numpy array"
        pipe = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=23)
        )
        pipe.fit(X_train, y_train)
        return pipe
    except AssertionError as msg:
        return msg


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    try:
        assert isinstance(y, np.ndarray), "Input y must be a Numpy array"
        assert isinstance(preds, np.ndarray), "Input preds must be a Numpy array"
        fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
        precision = precision_score(y, preds, zero_division=1)
        recall = recall_score(y, preds, zero_division=1)
        return precision, recall, fbeta
    except AssertionError as msg:
        return msg


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.linear_model._base.LinearRegression
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    try:
        pipeline = load(model)
    except:
        return 0
    lr_model = pipeline[1]
    logger.info(f"X.shape[1]: {X.shape[1]}")
    logger.info(f"type of lr_model: {type(lr_model)}")
    logger.info(f"lr_model.coef_: {lr_model.coef_}")
    logger.info(f"len(lr_model.coef_[0]): {len(lr_model.coef_[0])}")
    assert len(lr_model.coef_[0]) == X.shape[1], "Data dimension incorrect"
    preds = pipeline.predict(X)
    return preds
