import os
import pickle
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.utility import get_root_directory


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(get_root_directory(), "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        obj = DataTransformation()
        step1 = obj.data_transformer_pipeline()
        step2 = RandomForestRegressor(n_estimators=100,
                                      random_state=3,
                                      max_samples=0.5,
                                      max_features=0.75,
                                      max_depth=15)

        pipe = Pipeline([
            ('step1', step1),
            ('step2', step2)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        print('R2 score', r2_score(y_test, y_pred))
        print('MAE', mean_absolute_error(y_test, y_pred))
        pickle.dump(pipe, open(self.model_trainer_config.trained_model_file_path, 'wb'))


def run_train_pipeline():
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()
    print(train_data_path, test_data_path)
    data_transformation = DataTransformation()
    X_train, y_train, X_test, y_test = data_transformation.initiate_data_transformation(train_data_path,
                                                                                        test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)
