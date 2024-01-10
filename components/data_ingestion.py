import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
# from src.components.data_transformation import DataTransformation
from components.utility import get_root_directory


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(get_root_directory(), "train.csv")
    test_data_path: str = os.path.join(get_root_directory(), "test.csv")
    raw_data_path: str = os.path.join(get_root_directory(), "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
        df = pd.read_csv("/Users/shuchi/Documents/work/personal/python/house_price_prediction/data/realest.csv")

        for col in df.columns:
            if col in ["Space", "Tax", "Lot"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].median())

        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
        train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

        data_path = os.path.join(get_root_directory(), "data.pkl")
        pickle.dump(df, open(data_path, 'wb'))

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )