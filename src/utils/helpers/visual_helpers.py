"""This module contains functions to visualize data"""
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def cat_distribution(df: pd.DataFrame, column:str) -> Axes:
    """ Plot the distribution of a categorical variable """
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=column, palette="flare", hue=column, legend=False, stat="proportion")
    plt.title(column.capitalize() + ' Distribution')
    plt.xlabel(column)
    plt.xticks(rotation=30)
    plt.ylabel('Frequency')
    plt.show()


def num_distribution(df: pd.DataFrame, column:str) -> Axes:
    """ Plot the distribution of a numerical variable """
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(column.capitalize() + ' Distribution')
    plt.xlabel(column)
    plt.xticks(rotation=30)
    plt.ylabel('Frequency')
    plt.show()
