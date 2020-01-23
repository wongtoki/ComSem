from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dataloader import Loader
from sklearn.metrics import *

from model import *

import numpy
import pandas as pd


def search(features_comb, classifiers, data_train, data_test):

    columns = []
    rows = []

    dictionary = {}

    for features in features_comb:
        m = Model(data_train)
        feature_name = 0
        for feature in features[0]:
            m.add_feature(feature, "F" + str(feature_name))
            feature_name += 1

        for classifier in classifiers:

            m.train_model(classifier[0])

            acc = m.test_model(data_test, data_test["entailment_judgment"])
            comb_name = features[1]
            classifier_name = classifier[1]

            print(comb_name + " " + classifier_name)

            try:
                dictionary[comb_name].append([acc, classifier_name])
            except:
                dictionary[comb_name] = [[acc, classifier_name]]

    print(dictionary)
    result = pd.DataFrame(dictionary)
    result.to_csv("./search_result.csv")
    return dictionary


def test():

    data = Loader.load_data("../NLI2FOLI/SICK/SICK_train.txt")
    test = Loader.load_data("../NLI2FOLI/SICK/SICK_trial.txt")

    encoder = FeatureExtractor.generate_postag_onehot(data["tokens"])

    data["postags"] = FeatureExtractor.postag_tokenizer(data["tokens"])
    test["postags"] = FeatureExtractor.postag_tokenizer(test["tokens"])

    data["antons"] = FeatureExtractor.antonym_relations(data["pair_ID"])
    test["antons"] = FeatureExtractor.antonym_relations(test["pair_ID"])

    data["synons"] = FeatureExtractor.synonym_relations(
        data["tokens"], data["pair_ID"])
    test["synons"] = FeatureExtractor.synonym_relations(
        test["tokens"], test["pair_ID"])

    # Features
    # You don't have to do anything here
    # These are all the features we have

    # The tfidf features with both sentences seperated
    bag_of_words = ColumnTransformer([("A", TfidfVectorizer(), "sentence_A"),
                                      ("B", TfidfVectorizer(), "sentence_B")])

    # The countvectorizer features with postagging appended to each token
    bag_of_words_plus_pos = ColumnTransformer([("POS", CountVectorizer(
        tokenizer=lambda x:x, preprocessor=lambda x:x), "postags")])

    # The One-hot-encoded postag features for both sentences
    postags_A = POSTAGTransformer(encoder, "sentence_A", maxlen=800)
    postags_B = POSTAGTransformer(encoder, "sentence_B", maxlen=800)

    # The negation features for both sentences
    negation_A = NEGTransformer("sentence_A")
    negation_B = NEGTransformer("sentence_B")

    # The antonyms and synonyms features
    antons = DumbTransfromer("antons")
    synons = DumbTransfromer("synons")

    # classifiers
    # Every classifiers should be put in a tuple with its name on the right hand side
    # You may tweak the hyperparameters

    nb = (MultinomialNB(), "Naive Bayes")
    knn = (KNeighborsClassifier(), "KNN")
    svm = (SVC(kernel="linear"), "SVM")
    forest = (RandomForestClassifier(), "Random Forest")

    # Feature_combs
    # Feature combinations are a list of tuples
    # Give the combination a name on the right hand side
    combs = [
        ([bag_of_words], "TFIDF"),
        ([bag_of_words_plus_pos], "Combined + Postagging"),
        ([bag_of_words, postags_A, postags_B], "TFIDF + One hot postags"),
        ([bag_of_words, negation_A, negation_B], "TFIDF + NEGATION"),
        ([bag_of_words, antons, synons], "TFIDF + ANTONYMS + SYNONYMS"),
        ([bag_of_words, postags_B, postags_A, negation_A,
          negation_B, antons, synons], "All features")
    ]

    # The seach function takes the combination, classifier list and the train test data.
    # The result will be printed and exported to a csv file called search result
    # It also returns the dictionary of all the accuracys
    search(combs, [nb, knn, svm, forest], data, test)


if __name__ == "__main__":
    test()
