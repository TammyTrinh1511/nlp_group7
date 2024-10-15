import argparse
import logging
import warnings

import nltk
from nltk.corpus import twitter_samples

from src.problem_1_2.evaluate import ModelEvaluator
from src.problem_1_2.preprocess import Preprocessor
from src.problem_1_2.train import Trainer
from src.problem_1_2.tuning import Tuner

# Ignore all warnings
warnings.filterwarnings("ignore")

# Logging configuration
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


def str2bool(value):
    """
    Helper function to convert string input to boolean.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "t", "yes", "1"):
        return True
    elif value.lower() in ("false", "f", "no", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    """
    Parses command-line arguments for the pipeline runner.
    """
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline Runner")

    # Boolean argument for model tuning (True or False)
    parser.add_argument(
        "--tune",
        type=str2bool,
        default=False,
        help="Whether to tune the models. Options: True, False.",
    )

    # Argument to select which model to train
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help='Specify the model to train. Options: "Logistic Regression", "SVC", "Random Forest", \
            "Naive Bayes", "all".',
    )

    # Argument to evaluate a specific model or all models
    parser.add_argument(
        "--evaluate",
        type=str,
        default="all",
        help='Specify which model to evaluate. Use "--evaluate all" for evaluating all models\
            or specify a single model name.',
    )

    return parser.parse_args()


def main():
    # Parse the arguments
    """
    Main entry point for the pipeline runner.

    This function parses the command-line arguments, sets up the logging, and
    executes the pipeline steps in order (data ingestion, preprocessing, feature
    engineering, model training, model evaluation, and model tuning).

    :return: None
    """
    args = parse_args()

    logging.info("Pipeline execution started.")

    # # Load configuration from YAML file
    # with open("module/config.yaml", "r") as stream:
    #     config = yaml.safe_load(stream)

    # Step 1: Data Ingestion
    nltk.download("twitter_samples")
    all_positive_tweets_sen = twitter_samples.strings("positive_tweets.json")
    all_negative_tweets_sen = twitter_samples.strings("negative_tweets.json")

    logging.info("Data loading completed.")

    # Step 2: Preprocessing
    logging.info("Preprocessing started.")
    preprocessor = Preprocessor()
    train_x, test_x, y_train, y_test = preprocessor.preprocess_data(
        all_positive_tweets_sen, all_negative_tweets_sen
    )

    X_train = [" ".join(tokens) for tokens in train_x]
    X_test = [" ".join(tokens) for tokens in test_x]

    logging.info("Preprocessing finished.")

    # Step 3: Feature Engineering (if needed)
    logging.info("Feature engineering started.")
    logging.info("Feature engineering finished.")

    # Step 4: Model Training
    logging.info("Model training started.")
    trainer = Trainer()

    if args.model == "all":
        trainer.train_all_models(X_train, y_train, X_test, y_test)
    else:
        logging.info(f"Training only {args.model}.")
        trainer.train_model(args.model, X_train, y_train)

    logging.info("Model training completed.")

    # Step 5: Model Evaluation
    logging.info("Model evaluation started.")
    evaluator = ModelEvaluator()

    if args.evaluate == "all":
        logging.info("Evaluating all models.")
        for model_name in trainer.models.keys():
            errors = evaluator.error_analysis(
                trainer.train_model(model_name, X_train, y_train),
                X_test,
                y_test,
                test_x,
            )
            logging.info(f"Errors for {model_name}:")
            logging.info(f"\n{errors}")
    else:
        logging.info(f"Evaluating only {args.evaluate}.")
        errors = evaluator.error_analysis(
            trainer.train_model(args.evaluate, X_train, y_train), X_test, y_test, test_x
        )
        logging.info(f"Errors for {args.evaluate}:")
        logging.info(f"\n{errors}")

    logging.info("Model evaluation completed.")

    # Step 6: Model Tuning (if requested)
    if args.tune:
        logging.info("Model tuning started.")
        tuner = Tuner()

        # Example: Tuning Random Forest
        if args.model == "Random Forest" or args.model == "all":
            logging.info("Tuning Random Forest model.")
            rf_best_model, rf_best_params = tuner.tune_random_forest(X_train, y_train)
            logging.info(
                f"Random Forest tuning completed. Best parameters: {rf_best_params}"
            )

    logging.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()


# To run the tweet-pipeline-runner.py script, execute the following command:
# python tweet-pipeline-runner.py --model all --evaluate all --tune 0
