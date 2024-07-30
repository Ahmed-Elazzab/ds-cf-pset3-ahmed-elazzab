import logging
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split

from data_processor import DataProcessor
from predictors.lr_predictor import LinearModel
from predictors.svm_predictor import SVMModel
from utils import Utils

"""Add necessary imports."""

"""Create a dictionary that maps the string keys to corresponding model
classes. i.e LinearModel and SVM. these string should be respective of
what model we have defined in the config.yml file."""
PREDICTORS_LOOKUP = {"linear": LinearModel, "SVM": SVMModel}

""" Implement the configuration for logging in the project here. it should add a log file
in the output/logs/abc.log
Log file should contain logs of complete journey of training and predicting model.
The name of the log file should be like this. log-2023-06-06_200011.log"""
TIME_STAMP = datetime.now().strftime("%Y-%m-%d_%H%M%S")
log_filename = f"log-{TIME_STAMP}.log"
logging.basicConfig(filename=f"output/logs/{log_filename}", level=logging.INFO)

logging.info("Running Urban Planning")

logger = logging.getLogger("urbanGUI")


def orchestrate():
    """
    This is the entry point method responsible for all the functionality done in project.
    This method loads the configuration, reads the data from a file, and either trains a new model
    or loads a trained model for prediction.
    """
    # Create instance of util class
    utils = Utils()
    # Get the configuration from the util class with get_config method
    config = utils.get_config("config.yaml")
    if config is None:
        raise ValueError("Failed to load configuration.")
    # Load data from file using pandas read_csv method.
    dataframe = pd.read_csv(config["input_data"])
    # Create a predictor instance based on the model_type
    model_type = config["model_type"]
    if model_type in PREDICTORS_LOOKUP:
        predictor = PREDICTORS_LOOKUP[model_type]()
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    # Add check if train model bool is true in config. Then write steps to train model.
    # if config["train_model"]:
    # Train a new model using train method
    # train(utils, config, dataframe, model_type)
    # else:
    # Load a trained model using load_predict method
    # load_predict(utils, config, dataframe)
    if config["train_model"]:
        train(utils, config, dataframe, predictor)
    else:
        load_predict(utils, config, dataframe, predictor)


def train(utils, config, dataframe, model_type):
    """
    This method is used for training a new model
    """

    # Initialize the data processor and run it on the data
    processor = DataProcessor(config)

    # Create a new list 'traning_cols' which should contain  features
    # and target columns from the config.
    training_cols = pd.concat(
        [dataframe[config["data_params"]["features"]], dataframe[config["data_params"]["target"]]], axis=1
    )
    # create the dataframe with training columns only
    dataframe = pd.DataFrame(training_cols)

    # Run the preprocessing on the dataframe using data processing class.
    dataframe = processor.preprocess(dataframe)

    # Get the features columns from the dataframe and store it as X
    X = dataframe[config["data_params"]["features"]]
    # Get the target column from the dataframe and store it as Y
    Y = dataframe[config["data_params"]["target"]]
    # Split the data into train and test sets

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=config["test_size"], random_state=42)

    # Create a predictor instance based on the model_type i.e Linear or SVM Model.
    predictor = model_type
    # Train the model using fit method
    predictor.fit(X_train, y_train)
    # Evaluate the model with socre method
    score = predictor.score(X_test, y_test)
    logging.info(f"Model score: {score}")
    # generate a model saving path using util's class generate_model_path
    model_path = utils.generate_model_path(config, TIME_STAMP)

    # Save the model to the model path using util's save_pkl method
    model = {"processor": processor, "predictor": predictor}
    utils.save_pkl(model, model_path)


def load_predict(utils, config, dataframe):
    """Loads a trained model and predicts using it."""
    # This block of code is ran when predicting using a trained model
    # Load the model from the model path you can use utils load_pkl method
    model = utils.load_pkl(config["trained_model_file"])

    # get the trained processor and run it on the data
    processor = model["processor"]

    # Get the features columns from the dataframe and store it as X
    X = dataframe[config["data_params"]["features"]]
    # Ran the preprocessing on feature columns
    X = processor.pre_process(X)
    # get the trained predictor and predict using it
    predictor = model["predictor"]
    # Use the predict function to predict the for X
    y_pred = predictor.predict(X)
    # Create new dataframe with data=y_pred and index=X.index, columns=config["data_params"]["target"]
    final_df = pd.DataFrame(data=y_pred, index=X.index, columns=[config["data_params"]["target"]])
    # combine the predictions df  with the original data X
    final_df = pd.concat([X, final_df], axis=1)
    # inverse transform the predictions using data_processor's post_process function
    df = processor.post_process(final_df)

    # Save the predictions to a csv file using util's write_to_csv function
    utils.write_to_csv(df, config["predicted_result_path"])


if __name__ == "__main__":
    """
    Entry point of the script. Calls the run_model function.
    """
    orchestrate()
