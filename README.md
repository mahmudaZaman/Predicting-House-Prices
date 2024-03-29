# Predicting House Prices

## Problem Statement

The Predicting House Prices project is dedicated to creating a precise model for predicting house prices, utilizing critical features such as Bedroom count, Living space etc. This initiative aims to empower homebuyers, real estate professionals, and investors by providing them with valuable insights, enabling them to make well-informed decisions in the realm of property transactions.

## Dataset

In this project, I have employed the "Chicago House Price" dataset sourced from Kaggle, encompassing crucial information such as Price, Bedroom count, Living space, Room details, Lot specifications, Tax data, Bathroom count, and Garage information. The diverse range of features in this dataset forms the basis for training and validating our house price prediction model.

## Models

This project involved a detailed Exploratory Data Analysis (EDA) of the "Chicago House Price" dataset to understand, clean, and preprocess the data. Various machine learning models, including Linear Regression, Decision Tree Regressor, and RandomForest Regressor, were implemented to predict house prices. Notably, the RandomForest Regressor exhibited superior performance.

To enhance accessibility, a user-friendly House Prediction UI app was developed using Streamlit, leveraging the optimized RandomForest model. This app allows users to input key parameters and receive accurate predictions for house prices in Chicago suburbs, blending advanced machine learning with an intuitive interface for practical predictions.

## Amazon S3 for Data Management

The project efficiently stores and manages datasets in Amazon S3. This cloud-based storage solution guarantees data accessibility, scalability, and security.
Additionally, the trained model, serialized as a Pickle file, is stored on AWS S3. This not only streamlines model persistence but also facilitates its seamless integration into the Streamlit app. Loading the model directly from AWS S3 in the Streamlit app ensures real-time predictions with minimal latency.

## How to Get Started

To explore this project and its codebase, follow these steps:

1. Clone this repository to your local machine.
2. Review the project code and documentation to gain insights into the model development process and AWS integration.
   - pip install -r requirements.txt
   - streamlit run main.py
3. Feel free to reach out if you have any questions or are interested in collaborating further.

## Why This Project Matters

Accurate house price predictions are invaluable for individuals and professionals involved in real estate. The Predicting House Prices project underscores the potential of data-driven insights and AWS services to empower users with informed property-related decisions.

Thank you for your interest in this project. I believe that accurate property price predictions can significantly impact real estate transactions and investment strategies.
