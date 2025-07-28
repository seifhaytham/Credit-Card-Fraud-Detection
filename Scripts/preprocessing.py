import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    df = df.drop('id', axis=1)
    target = 'Class'
    corr = df.corr()
    target_corr = corr[target].abs()
    threshold = 0.1
    drop = target_corr[target_corr < threshold].index.tolist()
    if target in drop:
        drop.remove(target)
    df = df.drop(columns=drop)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df.drop(target, axis=1))
    return pd.DataFrame(scaled_data, columns=df.drop(target, axis=1).columns), df[target]