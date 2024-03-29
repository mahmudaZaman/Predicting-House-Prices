import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from config import app_config


@dataclass
class DataIngestionConfig:
    raw_data_uri: str = f"s3://{app_config.storage.bucket_name}/{app_config.storage.files.raw_data}"
    print("raw_data_uri", raw_data_uri)
    raw_data_pkl_uri: str = f"s3://{app_config.storage.bucket_name}/{app_config.storage.files.raw_data_pkl}"
    print("raw_data_pkl_uri", raw_data_pkl_uri)
    train_data_uri: str = f"s3://{app_config.storage.bucket_name}/{app_config.storage.files.train_data}"
    print("train_data_uri", train_data_uri)
    test_data_uri: str = f"s3://{app_config.storage.bucket_name}/{app_config.storage.files.test_data}"
    print("test_data_uri", test_data_uri)


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        print("self.ingestion_config ", self.ingestion_config)

    def initiate_data_ingestion(self):
        df = pd.read_csv(self.ingestion_config.raw_data_uri)

        for col in df.columns:
            if col in ["Space", "Tax", "Lot"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].median())

        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        df.to_csv(self.ingestion_config.raw_data_uri, index=False, header=True)
        train_set.to_csv(self.ingestion_config.train_data_uri, index=False, header=True)
        test_set.to_csv(self.ingestion_config.test_data_uri, index=False, header=True)

        df.to_pickle(self.ingestion_config.raw_data_pkl_uri)

        return (
            self.ingestion_config.train_data_uri,
            self.ingestion_config.test_data_uri
        )
