from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reporter=None):
        self.reporter = reporter

    def fit(self, X, y=None):
        fig = plot_image(X)
        if self.reporter:
            self.reporter.log("### Custom Transformation")
            self.reporter.log("Fitting the model with data...")
            self.reporter.log_plot(fig, "## Example Plot")
        return self

    def transform(self, X, y=None):
        # Transform logic
        X_transformed = X
        return X_transformed


def plot_image(df):
    # Logging a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(df['A'], df['B'])
    ax.set_title('Example Plot')
    return fig
    