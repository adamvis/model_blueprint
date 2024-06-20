import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_california_housing

class VIFEliminator(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=10.0, priority_order=None, report=False, analysis_only=False):
        self.threshold = threshold
        self.priority_order = priority_order
        self.report = report
        self.analysis_only = analysis_only
        self.features_to_keep_ = []
        self.elimination_report_ = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.features_to_keep_, self.elimination_report_ = self._eliminate_vif(X)
        return self

    def transform(self, X):
        if self.analysis_only:
            return X
        return X[:, self.features_to_keep_]

    def _eliminate_vif(self, X):
        dropped = True
        elimination_report = []
        current_features = X.columns.tolist()
        
        while dropped:
            dropped = False
            vif = pd.DataFrame()
            vif["feature"] = current_features
            vif["VIF"] = [variance_inflation_factor(X[current_features].values, i) for i in range(len(current_features))]

            max_vif = vif["VIF"].max()
            if max_vif > self.threshold:
                if self.priority_order is not None:
                    to_drop = [f for f in self.priority_order if f in vif.loc[vif["VIF"] > self.threshold, "feature"].values]
                    if to_drop:
                        feature_to_drop = to_drop[0]
                    else:
                        feature_to_drop = vif.sort_values(by="VIF", ascending=False)["feature"].iloc[0]
                else:
                    feature_to_drop = vif.sort_values(by="VIF", ascending=False)["feature"].iloc[0]

                current_features.remove(feature_to_drop)
                elimination_report.append((feature_to_drop, max_vif))
                dropped = True

        feature_indices = [X.columns.get_loc(col) for col in current_features]
        return feature_indices, elimination_report

    def get_elimination_report(self):
        if self.report:
            return pd.DataFrame(self.elimination_report_, columns=["Eliminated Feature", "VIF"])
        else:
            return "Report generation was not enabled. Set 'report=True' to enable it."

# Example usage
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    # Load the California housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X = pd.DataFrame(X, columns=housing.feature_names)

    # Define a priority order (optional)
    priority_order = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

    # Initialize and fit the transformer with reporting enabled and analysis_only mode
    vif_eliminator = VIFEliminator(threshold=10.0, priority_order=priority_order, report=True, analysis_only=False)
    vif_eliminator.fit(X)

    # Transform the data (only if not in analysis_only mode)
    X_transformed = vif_eliminator.transform(X.values)

    # Print the resulting features
    print("Selected features:", X.columns[vif_eliminator.features_to_keep_])

    # Print the elimination report
    print(vif_eliminator.get_elimination_report())
