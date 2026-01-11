"""Models package - exports model classes."""
from .naive_bayes import NaiveBayesModel
from .decision_tree import DecisionTreeModel
from .random_forest import RandomForestModel

MODELS = {
    'naive_bayes': NaiveBayesModel,
    'decision_tree': DecisionTreeModel,
    'random_forest': RandomForestModel,
}

MODEL_NAMES = list(MODELS.keys())
