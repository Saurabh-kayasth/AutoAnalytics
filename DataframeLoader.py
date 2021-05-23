import pandas as pd

class DataFrame_Loader():
    def __init__(self):
        print("Loadind DataFrame")

    def read_csv(self, data):
        self.df = pd.read_csv(data)
        return self.df