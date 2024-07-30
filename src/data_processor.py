import logging

import numpy
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataProcessor:
    """A class for performing data processing operations on a DataFrame."""

    def __init__(self, config):
        """
        Initialize the DataProcessor class.

        Args:
            config (dict): The configuration settings.
        """
        # initialize required variables...
        self.y_col = config["data_params"]["target"]
        self.num_cols = None
        self.cat_cols = None
        self.X_scaler = None
        self.y_scaler = None

    def fit(self, df):
        # Implement the training logic here
        # For example, calculate mean and std for normalization
        self.means = df.mean()
        self.stds = df.std()
        self.trained = True

    def preprocess(self, data):
        """
        Execute the data processing operations.

        Args:
            data (DataFrame): The DataFrame to process.
        """
        # 1. Filter out the numerical columns from the dataframe.
        self.num_cols = data.select_dtypes(include=numpy.number).columns.tolist()
        # 2. Filter out the categorical columns from the dataframe.
        self.cat_cols = data.select_dtypes(exclude=numpy.number).columns.tolist()
        # 3. Call remove null function on your data
        data = self.remove_nulls(data)
        # 4. Call remove duplicate function on your data
        data = self.remove_duplicates(data)
        # 5. Call remove standard Scale function on your data
        data = self.standard_scale(data)
        return data

    def post_process(self, data):
        """
        Execute the post-processing operations.

        Inverse the scaling operations the target column.
        The goal of this function is to make the output "human-readable".

        Args:
        data (DataFrame): The DataFrame to process.

        Steps:
          1. Implement the logic to inverse the standard scaling
             to generate output in "human-readable" format.
          2. return updated dataframe
        """
        # implementation for the logic inverse the standard scaling
        data[self.y_col] = self.y_scaler.inverse_transform(data[self.y_col].values.reshape(-1, 1))
        return data

    def remove_nulls(self, data):
        """
        Remove null values from specified columns in the DataFrame.

        Args:
            data (DataFrame): The DataFrame to process.

        Raises:
            KeyError: If the specified columns are not found in the DataFrame.
        """
        # the logic to remove null values
        null_less_data = data.dropna()

        return null_less_data

    def remove_duplicates(self, data):
        """
        Remove duplicate rows from the DataFrame.

        Args:
            data (pandas.DataFrame): The DataFrame to save.

        Raises:
            KeyError: If the specified columns are not found in the DataFrame.
        """
        # write logic to remove duplicate rows from all columns and return updated dataframe
        no_duplicates = data.drop_duplicates()
        return no_duplicates

    def standard_scale(self, data):
        """
        Scale the data using StandardScaler.

        Args:
            data (DataFrame): The DataFrame to scale.
        """
        # 1. Implement the logic to scale the features and target column to standard Scale.
        # 2. Keep in mind that you have to apply the same scale on prediction data
        # 3. return updated data

        # Initialize the scaler if not already initialized
        if self.X_scaler is None:
            self.X_scaler = StandardScaler()
            self.X_scaler.fit(data[self.num_cols])

        # Apply standard scaling to the numerical columns
        data[self.num_cols] = self.X_scaler.transform(data[self.num_cols])

        # Initialize the scaler for the target column if not already initialized
        if self.y_scaler is None:
            self.y_scaler = StandardScaler()
            self.y_scaler.fit(data[self.y_col])

        # Apply standard scaling to the target column
        data[self.y_col] = self.y_scaler.transform(data[self.y_col])

        return data

        #  IMPORTANT NOTE
        """Standard Scaling technique is used in data preprocessing to transform features by
        subtracting the mean and dividing by the standard deviation. This process standardizes
        the range of the features, making them have zero mean and unit variance. While this is
        true for training purpouses it's a different case for prediction purpouses. When predicting,
        the data must be transfromed using the same scale as the data for training, otherwise
        the we will be 'translating' data to a different 'language' (scale) and the model will
        make mistakes. Therefore when a model is being trained, a new scaler must be fit and 
        used to transform the data, but for predcition, the same scaler is used to only transform 
        the data."""
