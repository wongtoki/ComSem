from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dataloader import Loader
from sklearn.metrics import *

from model import *

import numpy


def search(features, classifiers, data_train, data_test):

    encoder = FeatureExtractor.generate_postag_onehot(data_train)

    for feature in features:
        for classfier in classfiers:
            m = Model(data)

    pass


def test():

    pass
