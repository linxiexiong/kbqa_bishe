import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBRegressor


class XGBoostModel(object):
    def __init__(self, max_depth=3, learning_rate=0.1, n_estimators=100):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

    def xgb_classifier(self):
        return XGBClassifier(max_depth=self.max_depth,
                             learning_rate=self.learning_rate,
                             n_estimators=self.n_estimators,
                             objective='binary:logistic')

    def xgb_regressor(self):
        return XGBRegressor(max_depth=self.max_depth,
                            learning_rate=self.learning_rate,
                            n_estimators=self.n_estimators,
                            objective='reg:linear')

    def xgb_ranker(self):
        return XGBClassifier(max_depth=self.max_depth,
                             learning_rate=self.learning_rate,
                             n_estimators=self.n_estimators,
                             objective='rank:pairwise')

class NumeralModel(object):
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
