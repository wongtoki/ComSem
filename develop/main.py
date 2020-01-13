from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dataloader import Loader
from sklearn.metrics import *

import numpy


class MeanEmbeddingVectorizer(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        # X should be data.vectorised
        vec = []
        for vectors in X:
            import numpy
            v = numpy.array(vectors)
            v = numpy.mean(v, axis=0)
            vec.append(v)

        return vec


class Model:
    def __init__(self, data, embeddings=None, vectorizer=CountVectorizer()):
        self.data = data
        self.vectorizer = vectorizer

        self.useWordEmbeddings = embeddings != None

        if self.useWordEmbeddings:
            embs = []
            for doc in self.data["tokens"]:
                vectors = []
                for token in doc:
                    try:
                        vectors.append(embeddings[token.lower()])
                    except KeyError:
                        # continue
                        vectors.append(embeddings['unk'])

                embs.append(vectors)

            self.data["embs"] = embs

    def justRun(self, test_size=0.2):
        """Split the data, train the data, and test the data in one line."""

        self.train, self.test = train_test_split(
            self.data, test_size=test_size)

        self.xtrain = self.train["tokens"]
        self.ytrain = self.train["entailment_judgment"]

        self.xtest = self.test["tokens"]
        self.ytest = self.test["entailment_judgment"]

        if self.useWordEmbeddings:
            self.xtrain = self.train["embs"]
            self.xtest = self.test["embs"]

        self.train_model()
        self.test_model()
        pass

    def train_model(self):
        classifier = Pipeline(
            [('vec', self.vectorizer), ('cls', SVC(kernel="sigmoid"))])
        classifier.fit(self.xtrain, self.ytrain)
        self.model = classifier

    def test_model(self):

        if self.model != None:
            pred = self.model.predict(self.xtest)
            print(classification_report(self.ytest, pred))
        else:
            print("Model not trained.")

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model


if __name__ == "__main__":

    from gensim.models import KeyedVectors

    print("loading word embeddings...")
    embs = KeyedVectors.load_word2vec_format(
        "../crawl-300d-2M.vec/crawl-300d-2M.vec")

    model = Model(Loader.load_data("../NLI2FOLI/SICK/SICK_train.txt"),
                  embeddings=embs,
                  vectorizer=MeanEmbeddingVectorizer())

    model.justRun()
