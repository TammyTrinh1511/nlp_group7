from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


class Trainer:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(),
            'SVC': SVC(),
            'Random Forest': RandomForestClassifier(),
            'Naive Bayes': MultinomialNB()
        }

    def train_model(self, model_name, X_train, y_train):
        model = self.models.get(model_name)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(), sublinear_tf=True)),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        return pipeline

    def train_all_models(self, X_train, y_train, X_test, y_test):
        for model_name in self.models:
            pipeline = self.train_model(model_name, X_train, y_train)
            y_pred = pipeline.predict(X_test)
            print(f"{model_name} Classification Report:")
            print(classification_report(y_test, y_pred))
