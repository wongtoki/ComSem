from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from dataloader import Loader


class Model:
    def __init__(self, data):
        self.train, self.test = train_test_split(data, test_size=0.2)
        print(len(data))
        print(len(self.test))
        pass


if __name__ == "__main__":
    model = Model(Loader.load_data("../NLI2FOLI/SICK/SICK_train.txt"))
