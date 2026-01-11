"""Naive Bayes Model."""
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesModel:
    name = "Naive Bayes"
    filename = "mnb.joblib"
    
    @staticmethod
    def get_model():
        return MultinomialNB()
