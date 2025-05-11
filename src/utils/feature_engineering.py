import pandas as pd 
import numpy as np

def create_features(df):
    # Weight_per_Age
    df['Weight_per_Age'] = df['Weight'] / (df['Age'] + 1)

    # HeartRate per Weight
    df['HeartRate_per_kg'] = df['Heart_Rate'] / df['Weight']

    # Duration Per Age
    df['Duration_per_age'] = df['Duration'] / (df['Age'] + 1)

    # Duration * Heart Rate
    df['Duration_heart_rate'] = df['Duration']*df['Heart_Rate']

    # Intensity
    df['Duration_per_weight'] = df['Duration']/df['Weight']

    # All Durations add and multi
    df['duration_sum'] = df['Duration_per_weight'] + \
        df['Duration_heart_rate']+df['Duration_per_age']
    df['duration_multi'] = df['Duration_per_weight'] * \
        df['Duration_heart_rate']*df['Duration_per_age']

    # Creating new column 'BMI'
    df['BMI'] = df['Weight']/(df['Height'] ** 2)
    df['BMI'] = df['BMI'].round(2)

    df['Body_Temp2'] = df['Body_Temp']**2

    numerical_features = [col for col in X.columns if col not in ["Sex","id", "Calories"]]

    categorical_features = ["Sex"]
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})

    for col in categorical_features:
        for num_feature in numerical_features:
            df[f'{num_feature}_x_{col}'] = df[num_feature] * df[col]


    return df

