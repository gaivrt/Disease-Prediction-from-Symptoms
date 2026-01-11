"""Decision Tree Model."""
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeModel:
    name = "Decision Tree"
    filename = "decision_tree.joblib"
    
    @staticmethod
    def get_model(criterion='entropy'):
        return DecisionTreeClassifier(criterion=criterion)
