"""Random Forest Model."""
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    name = "Random Forest"
    filename = "random_forest.joblib"
    
    @staticmethod
    def get_model(n_estimators=10):
        return RandomForestClassifier(n_estimators=n_estimators)
