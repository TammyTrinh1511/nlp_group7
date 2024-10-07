import pandas as pd


class ModelEvaluator:
    def error_analysis(self, pipeline, X_test, y_test, sentences):
        y_pred = pipeline.predict(X_test)
        df_errors = pd.DataFrame({
            'sentence': sentences,
            'predicted_class': y_pred,
            'real_class': y_test.flatten()
        })
        df_errors = df_errors[df_errors['predicted_class']
                              != df_errors['real_class']]
        return df_errors
