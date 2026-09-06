import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(BASE_DIR / "data" / "creditcard.csv")
    sample = df.drop('Class', axis=1).iloc[0]
    print(sample.to_json())


if __name__ == '__main__':
    main()
