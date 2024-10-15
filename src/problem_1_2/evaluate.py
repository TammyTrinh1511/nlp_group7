import pandas as pd


class ModelEvaluator:
    """Evaluate model"""

    def error_analysis(self, pipeline, X_test, y_test, sentences):
        """
        Perform error analysis on the pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to evaluate.
        X_test : array-like
            The test data.
        y_test : array-like
            The test labels.
        sentences : list of str
            The sentences corresponding to the test data.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the sentences, predicted labels, and real labels where the
            predicted labels and real labels do not match.
        """
        y_pred = pipeline.predict(X_test)
        df_errors = pd.DataFrame(
            {
                "sentence": sentences,
                "predicted_class": y_pred,
                "real_class": y_test.flatten(),
            }
        )
        df_errors = df_errors[df_errors["predicted_class"] != df_errors["real_class"]]
        return df_errors
