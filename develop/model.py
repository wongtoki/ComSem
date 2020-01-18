import pandas as pd
import random
import numpy as np
from dataloader import Loader
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import *
import nltk
from nltk.sentiment.util import mark_negation
from nltk.sentiment.vader import negated
from nltk.corpus import wordnet as wn


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

    def __init__(self):
        pass

    @staticmethod
    def postagger(data):
        new_data = []
        for tokens in data:
            new_tokens = []
            tags = nltk.pos_tag(tokens)
            for tag in tags:
                new_tokens.append(tag[0].lower())
                new_tokens.append(tag[1])

            new_data.append(new_tokens)
        return new_data

    @staticmethod
    def ccgtransformer(data):
        pass

    @staticmethod
    def negation_tagger(sentences):
        """Tags negation for list of tokens that comprises of a sentence

        :param list sentences: the premise or hypothesis
        :rtype: list
        :return: "_NEG" appended for tokens within negation's scope
        """
        return [mark_negation(sent) for sent in sentences]

    @staticmethod
    def bool_negation_tagger(sentences):
        """Tags negation for a sentence (not a list of tokens)

        :param list sentences: the premise or hypothesis
        :rtype: list
        :return: True for sentences that contain negation, otherwise False
        """
        tagged_data = []
        for sent in sentences:
            # preferably the sentences are not tokenized
            if isinstance(sent, str):
                tagged_data.append(negated(sent))

            elif isinstance(sent, list):
                sent = " ".join(sent) # if sentences are tokenized, join the tokens by whitespace
                tagged_data.append(negated(sent))

        return tagged_data


    @staticmethod
    def antonym_relations():
        pass

    @staticmethod
    def synonym_relations(sentences):
        pass


if __name__ == "__main__":
    data = Loader.load_data("../NLI2FOLI/SICK/SICK_train.txt")
    test = Loader.load_data("../NLI2FOLI/SICK/SICK_trial.txt")
    model = Model(data)

    model.x = FeatureExtractor.postagger(model.x)
    test_x = FeatureExtractor.postagger(test["tokens"])

    model.train_model()

    model.test_model(test_x, test["entailment_judgment"])
