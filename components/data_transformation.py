from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import os
from components.utility import save_object, get_root_directory


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(get_root_directory(), "proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def data_transformer_pipeline(self):
        ct = ColumnTransformer(transformers=[
            ('col_tnf', MinMaxScaler(), [0, 1, 2, 3, 4, 5, 6, 7])
        ], remainder='passthrough')
        return ct

    def initiate_data_transformation(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        target_column_name = "Price"

        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        y_train = train_df[target_column_name]
        y_train = y_train.to_numpy()

        input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
        y_test = test_df[target_column_name]
        y_test = y_test.to_numpy()

        X_train = input_feature_train_df
        X_test = input_feature_test_df

        return (
            X_train,y_train,
            X_test,y_test,
        )