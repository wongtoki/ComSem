from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from dataloader import Loader
from sklearn.metrics import *


class Model:
    def __init__(self, data):
        self.model = None
        self.train = data
        self.test = data
        pass

    def train_model(self):
        """Using a TFIDF vectorisor"""
        xtrain = self.train["tokens"]
        ytrain = self.train["entailment_judgment"]

        vec = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x, ngram_range=(1,1))

        classifier = Pipeline([('vec', vec), ('cls', LinearSVC(C=1))])
        classifier.fit(xtrain, ytrain)
        self.model = classifier

    def test_model(self):
        xtest = self.test["tokens"]
        ytest = self.test["entailment_judgment"]

        if self.model != None:
            pred = self.model.predict(xtest)
            print(classification_report(ytest, pred))
        else:
            print("Model not trained.")

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model


if __name__ == "__main__":
    model = Model(Loader.load_data("../NLI2FOLI/SICK/SICK_train.txt"))
    testModel = Model(Loader.load_data("../NLI2FOLI/SICK/SICK_trial.txt"))
    
    model.train_model()

    testModel.set_model(model.get_model())
    testModel.test_model()
    
