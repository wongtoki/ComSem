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
    def antonym_relations(sentences):
        wn_synsets = get_synsets("wordnet_synsets.csv")
        all_antonyms = []

        for pairID, sent in enumerate(sentences.tokens):
            pairID = str(pairID + 1)

            if pairID in wn_synsets.keys():
                all_synsets = wn_synsets[pairID]
                antonyms = []
                antonym_already_used = set()

                for ss in wn_synsets[str(pairID)]:
                    for lemma in ss.lemmas():
                        # antonym relations are captured in lemmas, not in synsets
                        for antonym_lemma in lemma.antonyms():
                            antonym = antonym_lemma.synset()  # return lemma form to synset form
                            if antonym in all_synsets and antonym not in antonym_already_used:
                                # only if the antonym occurs in the set of synsets form the formulas do we append it
                                # only if the antonym-relation has not already been added previously
                                antonym = (ss, antonym)
                                antonym_already_used.add(ss)  # add antonym so it cannot be re-used
                                antonyms.append(antonym)

                all_antonyms.append(antonyms)
            else:
                all_antonyms.append([])

        return all_antonyms

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
            x.append(mark_negation(tokens))
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

    model.x = FeatureExtractor.antonym_relations(model.x)
    test_x = FeatureExtractor.antonym_relations(test["tokens"])

    exit()


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

    # model.add_feature(POSTAGTransformer(encoder, "pos_A"), "PostagsA")
    # model.add_feature(POSTAGTransformer(encoder, "pos_B"), "PostagsB")

    model.train_model()

    model.test_model(test, test["entailment_judgment"])
