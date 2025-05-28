import pandas as pd
import numpy as np


def add_cross_terms(df, features):
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            df[f"{features[i]}_x_{features[j]}"] = df[features[i]] * \
                df[features[j]]
    return df


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
    numerical = ["Age", "Height", "Weight",
                 "Duration", "Heart_Rate", "Body_Temp"]
    df = add_cross_terms(df, numerical)

    df['Sex'] = df['Sex'].map({'female': 1, 'male': 0})
    df['Sex_Reversed'] = 1 - df['Sex']
    for f1 in ['Duration', 'Heart_Rate', 'Body_Temp']:
        for f2 in ['Sex', 'Sex_Reversed']:
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]

    for age in df['Age'].unique():
        df[f'Heart_Rate_Age_{int(age)}'] = np.where(
            df['Age'] == age, df['Heart_Rate'], 0)
        df[f'Body_Temp_Age_{int(age)}'] = np.where(
            df['Age'] == age, df['Body_Temp'], 0)

    for dur in df['Duration'].unique():
        df[f'Heart_Rate_Duration_{int(dur)}'] = np.where(
            df['Duration'] == dur, df['Heart_Rate'], 0)
        df[f'Body_Temp_Duration_{int(dur)}'] = np.where(
            df['Duration'] == dur, df['Body_Temp'], 0)

    df.drop(columns=['max_heart_rate', 'Duration_heart_rate'])
    
    return df
