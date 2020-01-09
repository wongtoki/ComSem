import pandas as pd


class Loader:
    @staticmethod
    def load_data(path: str):
        """The path should a txt file containing sick data"""

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
                    for x, c in enumerate(contents):
                        data[labels[x]].append(c)

                i += 1

        return pd.DataFrame(data)


if __name__ == "__main__":
    path = "../NLI2FOLI/SICK/SICK_train.txt"
    data = Loader.load_data(path)
    print(data)
