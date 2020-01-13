import pandas as pd
import random
import numpy as np
from dataloader import Loader
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
import nltk


class Model:
    def __init__(self, train_data):
        self.x = train_data["tokens"]
        self.y = train_data["entailment_judgment"]

    def train_model(self):

        pipe = Pipeline([('vec', CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)),
                         ('cls', RandomForestClassifier(n_estimators=100))])

        pipe.fit(self.x, self.y)
        self.model = pipe

    def test_model(self, x, y):
        self.prediction = self.model.predict(x)
        print(classification_report(y, self.prediction))


class FeatureExtractor:

    @staticmethod
    def postagger(data):
        new_data = []
        for tokens in data:
            new_tokens = []
            tags = nltk.pos_tag(tokens)
            for tag in tags:
                new_tokens.append(str(tag[0]+"_"+tag[1]).lower())

            new_data.append(new_tokens)
        return new_data

    @staticmethod
    def ccgtransformer(data):
        pass


if __name__ == "__main__":
    data = Loader.load_data("../NLI2FOLI/SICK/SICK_train.txt")
    test = Loader.load_data("../NLI2FOLI/SICK/SICK_trial.txt")
    model = Model(data)

    model.x = FeatureExtractor.postagger(model.x)
    test_x = FeatureExtractor.postagger(test["tokens"])

    model.train_model()

    model.test_model(test_x, test["entailment_judgment"])
