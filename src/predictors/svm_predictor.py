"""A class for SVM model that uses MLPredictor abstract class."""
from predictors.abstract_predictor import PredictorABC
from sklearn.svm import SVR


class SVMModel(PredictorABC):
    """SVM model class using abstract model class."""

    def fit(self, X_train, y_train):
        """Builds the ml model on train data."""
        # 1. Assign the model property of Parent predictor to SVR
        # 2. Apply fit method to train model accordingly
        pass
    def predict(self, X):
        """Applies the model to predict price for given X."""
        pass
