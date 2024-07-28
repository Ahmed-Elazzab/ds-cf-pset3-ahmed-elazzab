import logging
from datetime import datetime

import pandas as pd
from data_processor import DataProcessor
from predictors.lr_predictor import LinearModel
from predictors.svm_predictor import SVMModel
from sklearn.model_selection import train_test_split
from utils import Utils

"""Add necessary imports."""

"""Create a dictionary that maps the string keys to corresponding model
classes. i.e LinearModel and SVM. these string should be respective of
what model we have defined in the config.yml file."""
PREDICTORS_LOOKUP = {...}

""" Implement the configuration for logging in the project here. it should add a log file
in the output/logs/abc.log
Log file should contain logs of complete journey of training and predicting model.
The name of the log file should be like this. log-2023-06-06_200011.log"""
TIME_STAMP = ...
log_filename = ...
logging.basicConfig(...)

logging.info("Running Urban Planning")

logger = logging.getLogger("urbanGUI")


def orchestrate():
    """
    This is the entry point method responsible for all the functionality done in project.
    This method loads the configuration, reads the data from a file, and either trains a new model
    or loads a trained model for prediction.
    """
    # Create instance of util class

    # Get the configuration from the util class with get_config method

    # Load data from file using pandas read_csv method.

    # Create a predictor instance based on the model_type

    # Add check if train model bool is true in config. Then write steps to train model.

    # if config["train_model"]:
    # Train a new model using train method
    # train(utils, config, dataframe, model_type)
    # else:
    # Load a trained model using load_predict method
    # load_predict(utils, config, dataframe)


def train(utils, config, dataframe, model_type):
    """
    This method is used for training a new model
    """

    # Initialize the data processor and run it on the data

    # Create a new list 'traning_cols' which should contain  features
    # and target columns from the config.

    # create the dataframe with training columns only

    # Run the preprocessing on the dataframe using data processing class.

    # Get the features columns from the dataframe and store it as X

    # Get the target column from the dataframe and store it as Y

    # Split the data into train and test sets

    # Create a predictor instance based on the model_type i.e Linear or SVM Model.

    # Train the model using fit method

    # Evaluate the model with socre method

    # generate a model saving path using util's class generate_model_path
    # model_path = utils.generate_model_path(config, TIME_STAMP)

    # Save the model to the model path using util's save_pkl method
    # model = {"processor": processor, "predictor": predictor}
    # utils.save_pkl(model, model_path)


def load_predict(utils, config, dataframe):
    """Loads a trained model and predicts using it."""
    # This block of code is ran when predicting using a trained model
    # Load the model from the model path you can use utils load_pkl method
    # model = utils.load_pkl(model_path)

    # get the trained processor and run it on the data
    # processor = model["processor"]

    # Get the features columns from the dataframe and store it as X

    # Ran the preprocessing on feature columns

    # get the trained predictor and predict using it

    # Use the predict function to predict the for X

    # Create new dataframe with data=y_pred and index=X.index, columns=config["data_params"]["target"]

    # combine the predictions df  with the original data X

    # inverse transform the predictions using data_processor's post_process function
    # df = processor.post_process(final_df)

    # Save the predictions to a csv file using util's write_to_csv function
    # utils.write_to_csv(df, config["predicted_result_path"])


if __name__ == "__main__":
    """
    Entry point of the script. Calls the run_model function.
    """
    orchestrate()
