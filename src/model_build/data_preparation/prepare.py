import os
import requests
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)


def prepare_data(path='data'):
    # GET CENSUS DATASET
    logging.info("GET DATASET")

    urls = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    )

    if not os.path.exists(path):
        os.mkdir(path)

    for url in urls:
        response = requests.get(url)
        name = os.path.basename(url)
        logging.info(f"Processing {name}")
        with open(os.path.join(path, name), 'wb') as f:
            f.write(response.content)

    names = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education-num',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
        'native-country',
        'income',
    ]

    data = pd.read_csv('data/adult.data', names=names, skipinitialspace = True)
    logging.info("data.replace('?', np.nan)")
    for col in names:
        data[col] = data[col].apply(lambda x: np.nan if x == '?' else x)

    data.dropna(inplace=True)
    # SAVE DATASET INTO CSV FILE
    logging.info("SAVE DATASET INTO CSV FILE")
    output_folder = os.path.join(path, "census.csv")
    data.to_csv(output_folder, sep=",", index=False)

if __name__ == "__main__":
    prepare_data()
