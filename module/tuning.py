from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import lightgbm as lgb


class Tuner:
    def tune_random_forest(self, X_train, y_train):
        rf_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(), sublinear_tf=True)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        rf_param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__bootstrap': [True, False]
        }
        grid_search = GridSearchCV(
            rf_pipeline, rf_param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_

    def tune_lightgbm(self, X_train, y_train):
        lgb_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(tokenizer=lambda x: x.split(), sublinear_tf=True)),
            ('classifier', lgb.LGBMClassifier(random_state=42))
        ])
        lgb_param_grid = {
            'classifier__num_leaves': np.arange(20, 100, 5),
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__n_estimators': [50, 100, 200]
        }
        grid_search = GridSearchCV(
            lgb_pipeline, lgb_param_grid, cv=3, scoring='accuracy', verbose=2)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_params_
