"""Base class for all the predictors.

The parameters (attributes) are initialized by the child model class.
"""
import logging
from abc import ABC, abstractmethod

from sklearn import metrics
from statsmodels.stats.stattools import durbin_watson


logger = logging.getLogger(__name__)


class PredictorABC(ABC):
    """Abstract class for all Predictor types."""

    def __init__(self):
        """Initialize the predictor."""
        self.model = None

    @abstractmethod
    def fit(self, X_train, y_train):
        """Fit predictor.

        X:{ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples.
        y:ndarray of shape (n_samples,)
        The input target value.

        self : object
        Fitted estimator.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X):
        """Applies the model to predict price for given X."""
        raise NotImplementedError()

    def score(self, X_test, y_test):
        """
        X:{ndarray, sparse matrix} of shape (n_samples, n_features)
        Test samples.
        y:ndarray of shape (n_samples,)
        True values for X.

        score:float
        """
        # 1. Implement a method the return the MAPE, R2 and Durbin-Watson of the model. It should be dictionary
        # score = {
        #     "MAPE": mape,
        #     "R2": r2,
        #     "Durbin-Watson": dw,
        # }
        # return score