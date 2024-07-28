"""A class for Linear Regression model that uses MLPredictor abstract class."""

from sklearn.linear_model import LinearRegression

from predictors.abstract_predictor import PredictorABC


class LinearModel(PredictorABC):
    """Linear model class using abstract model class."""

    def fit(self, X_train, y_train):
        """Builds the ml model on train data."""
        # 1. Assign the model property of Parent predictor to LinearRegression
        # 2. Apply fit method to train model accordingly
        pass

    def predict(self, X):
        """Applies the model to predict price for given X."""
        pass
