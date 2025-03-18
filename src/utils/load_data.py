import os
import pandas as pd
from zipfile import ZipFile
from tensorflow import keras

def load_data():
    uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

    # Download ZIP file into `../data/` (without creating `dataset/` folder)
    zip_path = keras.utils.get_file(
        origin=uri,
        cache_dir="../data",  
        cache_subdir=".",  # Prevents the `dataset/` folder from being created
        fname="jena_climate_2009_2016.csv.zip"
    )

    # Extract the CSV file into `../data/`
    with ZipFile(zip_path, 'r') as zip_file:
        zip_file.extractall(path="../data")  

    # Define CSV path
    csv_path = "../data/jena_climate_2009_2016.csv"

    # Remove the ZIP file after extraction
    os.remove(zip_path)

    # Load the CSV file into a Pandas DataFrame
    df = pd.read_csv(csv_path)

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.info())
