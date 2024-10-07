import logging
import yaml
import warnings
import nltk
from nltk.corpus import twitter_samples
from module.preprocess import Preprocessor
from module.train import Trainer
from module.evaluate import ModelEvaluator
from module.tuning import Tuner

# Ignore all warnings
warnings.filterwarnings("ignore")

# Load configuration from YAML file
with open('module/config.yaml', 'r') as stream:
    # Using safe_load to avoid unsafe loading warnings
    config = yaml.safe_load(stream)

# Logging configuration
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)


def main():
    logging.info("Pipeline execution started.")

    # Step 1: Data Ingestion
    nltk.download('twitter_samples')
    all_positive_tweets_sen = twitter_samples.strings('positive_tweets.json')
    all_negative_tweets_sen = twitter_samples.strings('negative_tweets.json')

    logging.info("Data loading completed.")

    # Step 2: Preprocessing
    logging.info("Preprocessing started.")
    preprocessor = Preprocessor()
    train_x, test_x, y_train, y_test = preprocessor.preprocess_data(
        all_positive_tweets_sen, all_negative_tweets_sen)

    X_train = [' '.join(tokens) for tokens in train_x]
    X_test = [' '.join(tokens) for tokens in test_x]

    logging.info("Preprocessing finished.")

    # Step 3: Feature Engineering (if needed)
    logging.info("Feature engineering started.")
    # (You can add feature engineering steps here if needed)
    logging.info("Feature engineering finished.")

    # Step 4: Model Training
    logging.info("Model training started.")
    trainer = Trainer()
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    logging.info("Model training completed.")

    # Step 5: Model Evaluation
    logging.info("Model evaluation started.")
    evaluator = ModelEvaluator()
    logistic_errors = evaluator.error_analysis(trainer.train_model(
        'Logistic Regression', X_train, y_train), X_test, y_test, test_x)
    logging.info("Model evaluation completed. Errors for Logistic Regression:")
    logging.info(f"\n{logistic_errors}")

    # # Step 6: Model Tuning (optional)
    # logging.info("Model tuning started.")
    # tuner = Tuner()
    # rf_best_model, rf_best_params = tuner.tune_random_forest(X_train, y_train)
    # logging.info(
    #     f"Random Forest tuning completed. Best parameters: {rf_best_params}")

    logging.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
