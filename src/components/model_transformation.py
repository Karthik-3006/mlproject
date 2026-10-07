import pandas as pd
import os
import sys
import numpy as np
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object

class DataTransformationConfig:
    preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformer:
    def __init__(self):
        self.preprocessor_path = DataTransformationConfig()
    def data_transformer(self):
        try:
            num_features = ["math score", "reading score"]
            cat_features = ["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course"]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencode", OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("numpipeline", num_pipeline, num_features),
                    ("catpipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read Train and test data")
            logging.info("Getting preprocessor object")
            preprocessor_obj = self.data_transformer()
            target_col = "writing score"
            num_features = ["math score", "reading score"]
            input_xtrain_df = train_df.drop(target_col, axis=1)
            target_ytrain_df = train_df[target_col]

            input_xtest_df = test_df.drop(target_col, axis=1)
            target_ytest_df = test_df[target_col]

            input_xtrain_arr = preprocessor_obj.fit_transform(input_xtrain_df)
            input_xtest_arr = preprocessor_obj.transform(input_xtest_df)
            train_arr = np.c_[
                input_xtrain_arr,
                np.array(target_ytrain_df)
            ]

            test_arr = np.c_[
            input_xtest_arr,
            np.array(target_ytest_df)
            ]

            save_object(
                file_path=self.preprocessor_path.preprocessor_file_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.preprocessor_path.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)