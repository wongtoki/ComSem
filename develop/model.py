import pandas as pd
import random
import numpy as np
from dataloader import Loader
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
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

    def train_model(self, algo):
        features = FeatureUnion(self.transformers)
        pipe = Pipeline([('features', features),
                         ('cls', algo)])

        pipe.fit(self.x, self.y)
        self.model = pipe

    def test_model(self, x, y):
        self.prediction = self.model.predict(x)
        print(classification_report(y, self.prediction))


class FeatureExtractor:

    @staticmethod
    def postag_tokenizer(x, nosplit=True):
        X = []
        for sentence in x:
            if not nosplit:
                sentence = sentence.split()

            sent = nltk.pos_tag(sentence)
            tokens = []
            for token in sent:
                tokens.append(token[0]+"_"+token[1])
            X.append(tokens)

        return X

    @staticmethod
    def negation_tagger(sentence):
        """Tags negation for list of tokens that comprises of a sentence

        :param list sentences: the premise or hypothesis
        :rtype: list
        :return: "_NEG" appended for tokens within negation's scope
        """
        return mark_negation(sentence)

    @staticmethod
    def bool_negation_tagger(sentence):
        """Tags negation for a sentence (not a list of tokens)

        :param list sentences: the premise or hypothesis
        :rtype: list
        :return: True for sentences that contain negation, otherwise False
        """

        return negated(sentence)

    @staticmethod
    def has_antonyms(pairID, wn_synsets):

        if pairID in wn_synsets.keys():
            all_synsets = wn_synsets[pairID]

            for ss in wn_synsets[str(pairID)]:
                for lemma in ss.lemmas():
                    # antonym relations are captured in lemmas, not in synsets
                    for antonym_lemma in lemma.antonyms():
                        antonym = antonym_lemma.synset()  # return lemma form to synset form
                        if antonym in all_synsets:
                            # only if the antonym occurs in the set of synsets form the formulas do we append it
                            return [1]
                return [0]
        else:
            return [0]

    @staticmethod
    def antonym_relations(pairIDs):
        wn_synsets = get_synsets("wordnet_synsets.csv")

        vectors = []
        for pairID in pairIDs:
            b = FeatureExtractor.has_antonyms(pairID, wn_synsets)
            vectors.append(b)

        return vectors

    @staticmethod
    def grenerate_aton_data(data):
        relations = []
        for ID in data["pair_ID"]:
            relation = FeatureExtractor.antonym_relations(ID)
            relations.append(relation)

        return relations

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


class POSTAGTransformer(object):
    def __init__(self, encoder, column, maxlen=1000, nosplit=False):
        self.encoder = encoder
        self.maxlen = maxlen
        self.nosplit = nosplit
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        vecs = []

        for tokens in X[self.column]:
            if not self.nosplit:
                tokens = tokens.split()

            tagged_tokens = nltk.pos_tag(tokens)
            tokens = []
            for tagged_token in tagged_tokens:
                tokens.append(tagged_token[1])

            vectors = []
            for tag in tokens:
                vec = self.encoder.transform([[tag]]).toarray()
                vectors.extend(vec[0])

            diff = self.maxlen - len(vectors)
            if diff > 0:
                for n in range(diff):
                    vectors.append(0)

            elif diff < 0:
                for n in range(abs(diff)):
                    vectors.pop()

            vecs.append(vectors)

        return vecs


class NEGTransformer(object):
    def __init__(self, column, maxlen=50):
        self.column = column
        self.maxlen = maxlen

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        maxobj = max(X[self.column], key=len)

        # maxlen = len(maxobj.split())
        maxlen = self.maxlen

        vecs = []
        for tokens in X[self.column]:
            vectors = []
            negs = FeatureExtractor.negation_tagger(tokens.split())

            for n in negs:
                if str(n).endswith("NEG"):
                    vectors.append(1)
                else:
                    vectors.append(0)

            if negated(tokens.split()):
                vectors.append(3)
            else:
                vectors.append(4)

            diff = maxlen - len(vectors)
            if diff > 0:
                for n in range(diff):
                    vectors.append(2)
            elif diff < 0:
                for n in range(abs(diff)):
                    vectors.pop()

            vecs.append(vectors)
        return vecs


class DumbTransfrom(object):
    def __init__(self, column):
        self.column = column
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


def get_synsets(file):
    """Create dictionary from .csv file with synsets per problem number

    :param str: .csv file containing the synset stings
    :rtype: dict
    :return: dictionary w/ synsets in a list per problem nr
    """
    wordnet_ss = {}

    with open(file, "r") as ss_file:
        for line in ss_file:
            line = line.split("|")
            nr = line[0]
            synsets = line[1].rstrip().split(",")[:-1]
            wn_synsets = []
            for ss in synsets:
                ss = wn.synset(ss)
                wn_synsets.append(ss)

            wordnet_ss[nr] = wn_synsets

    return wordnet_ss


if __name__ == "__main__":
    data = Loader.load_data("../NLI2FOLI/SICK/SICK_train.txt")
    test = Loader.load_data("../NLI2FOLI/SICK/SICK_trial.txt")

    model = Model(data)

    print(FeatureExtractor.antonym_relations(data["pair_ID"]))

    exit()

    encoder = FeatureExtractor.generate_postag_onehot(data["tokens"])

    data["postags"] = FeatureExtractor.postag_tokenizer(data["tokens"])
    test["postags"] = FeatureExtractor.postag_tokenizer(test["tokens"])

    transformer = ColumnTransformer(
        [("A", TfidfVectorizer(), "sentence_A"),
         ("B", TfidfVectorizer(), "sentence_B"),
         ("POS", CountVectorizer(tokenizer=lambda x:x, preprocessor=lambda x:x), "postags")])

    model.add_feature(transformer, "bags")
    model.add_feature(POSTAGTransformer(encoder, "sentence_A"), "pos_a")
    model.add_feature(POSTAGTransformer(encoder, "sentence_B"), "pos_b")
    model.add_feature(NEGTransformer("sentence_A"), "neg_A")
    model.add_feature(NEGTransformer("sentence_B"), "neg_B")

    model.train_model(MultinomialNB())
    model.test_model(test, test["entailment_judgment"])
