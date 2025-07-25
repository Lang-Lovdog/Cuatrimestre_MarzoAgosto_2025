import pandas as pd
import sys

if __name__ == "__main__":
    csvFile = sys.argv[1]
    df = pd.read_csv(csvFile)
    # Set category column to 0 | 1 instead of M | R
    df['category'] = df['category'].map({'M': 0, 'R': 1})
    df.to_csv(csvFile[:-4]+"_dummy.csv", index=False)
