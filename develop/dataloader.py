import pandas as pd


class Loader:
    @staticmethod
    def load_data(path: str):
        """The path should a txt file containing sick data"""
        print("Loading data...")
        data = {}

        with open(path, 'r') as file:
            i = 0
            labels = []
            for row in file:
                if i == 0:
                    labels = str(row).strip().split('\t')

                    for l in labels:
                        data[l] = []

                else:
                    contents = str(row).strip().split('\t')
                    for n in range(len(labels)):
                        try:
                            data[labels[n]].append(contents[n])
                        except IndexError:
                            data[labels[n]].append("None")

                i += 1

        data["tokens"] = []
        for x, sent in enumerate(data["sentence_A"]):
            tokens = str(sent) + " " + data["sentence_B"][x]
            tokens = tokens.split()
            data["tokens"].append(tokens)

        return pd.DataFrame(data)


if __name__ == "__main__":
    path = "../NLI2FOLI/SICK/SICK_train.txt"
    data = Loader.load_data(path)
    print(data["tokens"][0])
