from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


class Trainer:
    """Train model"""

    def __init__(self):
        """
        Initialize the Trainer instance with a dictionary of models.

        The `models` dictionary contains the following models:
            - Logistic Regression
            - SVC
            - Random Forest
            - Naive Bayes

        Attributes:
            models (Dict[str, Any]): A dictionary of models.
        """
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "SVC": SVC(),
            "Random Forest": RandomForestClassifier(),
            "Naive Bayes": MultinomialNB(),
        }

    def train_model(self, model_name, X_train, y_train):
        """
        Train a model with given name using the input data.

        Parameters
        ----------
        model_name : str
            The name of the model to train. Must be one of the models in self.models.
        X_train : List[str]
            The training data.
        y_train : List[int]
            The labels of the training data.

        Returns
        -------
        Pipeline
            A pipeline containing the TF-IDF vectorizer and the trained model.
        """
        model = self.models.get(model_name)
        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(tokenizer=lambda x: x.split(), sublinear_tf=True),
                ),
                ("classifier", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        return pipeline

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models in self.models using the input data and print the classification report
        of each model.

        Parameters
        ----------
        X_train : List[str]
            The training data.
        y_train : List[int]
            The labels of the training data.
        X_test : List[str]
            The testing data.
        y_test : List[int]
            The labels of the testing data.
        """
        for model_name in self.models:
            pipeline = self.train_model(model_name, X_train, y_train)
            y_pred = pipeline.predict(X_test)
            print(f"{model_name} Classification Report:")
            print(classification_report(y_test, y_pred))
