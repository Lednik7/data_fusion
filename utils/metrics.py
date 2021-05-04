from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer


class F1WeightedScorer:

    def __init__(self, class_weights):
        self.class_weights = class_weights

    def __call__(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average=None).dot(self.class_weights)


f1_weighted_sc = make_scorer(F1WeightedScorer)