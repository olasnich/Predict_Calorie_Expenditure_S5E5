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

    df['max_heart_rate'] = 220 - df['Age']
    df['METs'] = (df['Heart_Rate'] / df['max_heart_rate']) * 8.8
    df['VO2_max'] = (df['Heart_Rate'] / (df['Age'] + 10)) * 100
    df['activity_score'] = (df['Duration'] * 2) + \
        (df['Heart_Rate'] * 3) + (df['Body_Temp'] * 1)

    categorical_features = ["Sex"]
    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df.drop(columns=['max_heart_rate'])
    
    return df
