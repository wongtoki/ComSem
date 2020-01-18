import pandas as pd
import random
import numpy as np
from dataloader import Loader
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import *
import nltk
from nltk.sentiment.util import mark_negation
from nltk.sentiment.vader import negated
from nltk.corpus import wordnet as wn


class Model:
    def __init__(self, train_data):
        self.x = train_data
        self.y = train_data["entailment_judgment"]

        self.transformers = []

    def add_feature(self, transformer, feature_name):
        self.transformers.append((feature_name, transformer))

    def train_model(self):
        features = FeatureUnion(self.transformers)
        pipe = Pipeline([('features', features),
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
    def add_pos_tags(sentence):
        print(nltk.help.upenn_tagset())

        sentence = sentence.split()
        sent = nltk.pos_tag(sentence)
        tokens = []
        for token in sentence:
            tokens.append()

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
                # if sentences are tokenized, join the tokens by whitespace
                sent = " ".join(sent)
                tagged_data.append(negated(sent))

        return tagged_data

    @staticmethod
    def antonym_relations():
        pass

    @staticmethod
    def synonym_relations(sentences):
        pass


class Helper:
    @staticmethod
    def generate_postag_onehot(documents):
        X = []
        for tokens in documents:
            tagged = nltk.pos_tag(tokens)
            for token in tagged:
                X.append([token[1]])

        encoder = OneHotEncoder()
        encoder.fit(X)
        return encoder

    def generate_postag_data(documents):
        x = []
        for tokens in documents:

            tags = []
            tagged = nltk.pos_tag(tokens.split())
            for token in tagged:
                tags.append(token[1])

            x.append(tags)
        return x

    def generate_negation_tags(documents):
        x = []
        for tokens in documents:
            x.append(mark_negation(tokens.split()))
        return x


class POSTAGTransformer(object):
    def __init__(self, encoder, column_name, length=1000):
        self._encoder = encoder
        self._column_name = column_name
        self._length = length

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        vecs = []

        for tokens in X[self._column_name]:
            vectors = []
            for tag in tokens:
                vec = encoder.transform([[tag]]).toarray()
                vectors.extend(vec[0])

            diff = self._length - len(vectors)
            if diff > 0:
                for n in range(diff):
                    vectors.append(0)

            elif diff < 0:
                for n in range(abs(diff)):
                    vectors.pop()

            vecs.append(vectors)

        return vecs


if __name__ == "__main__":
    data = Loader.load_data("../NLI2FOLI/SICK/SICK_train.txt")
    test = Loader.load_data("../NLI2FOLI/SICK/SICK_trial.txt")

    model = Model(data)

    encoder = Helper.generate_postag_onehot(data["tokens"])
    data["pos_A"] = Helper.generate_postag_data(data["sentence_A"])
    data["pos_B"] = Helper.generate_postag_data(data["sentence_B"])

    test["pos_A"] = Helper.generate_postag_data(test["sentence_A"])
    test["pos_B"] = Helper.generate_postag_data(test["sentence_B"])

    data["neg_A"] = Helper.generate_negation_tags(data["sentence_A"])
    data["neg_B"] = Helper.generate_negation_tags(data["sentence_B"])

    test["neg_A"] = Helper.generate_negation_tags(test["sentence_A"])
    test["neg_B"] = Helper.generate_negation_tags(test["sentence_B"])

    transformer = ColumnTransformer(
        [("A", CountVectorizer(), "sentence_A"),
         ("B", CountVectorizer(), "sentence_B"),
         ("NegA", CountVectorizer(tokenizer=lambda x: x,
                                  preprocessor=lambda x: x), "neg_A"),
         ("NegB", CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x), "neg_B")])

    model.add_feature(transformer, "Sentences")

    model.add_feature(POSTAGTransformer(encoder, "pos_A"), "PostagsA")
    model.add_feature(POSTAGTransformer(encoder, "pos_B"), "PostagsB")

    model.train_model()

    model.test_model(test, test["entailment_judgment"])
