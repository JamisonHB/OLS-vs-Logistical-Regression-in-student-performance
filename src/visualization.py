import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_studytime_vs_grade(df):
    sns.boxplot(x="studytime", y="G3", data=df)
    plt.title("Study Time vs Final Grade")
    plt.show()

def plot_absences(df):
    sns.scatterplot(x="absences", y="G3", data=df, alpha=.6)
    plt.title("Absences vs Final Grade")
    plt.show()

def plot_correlation(df):
    numeric_df = df.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_pairplot(df):
    subset = df[["studytime", "failures", "absences", "G3"]]
    sns.pairplot(subset, y_vars="G3", x_vars=subset.columns[:-1], kind="reg")
    plt.show()

def plot_grade_distribution(df):
    sns.histplot(df["G3"], bins=20, kde=True)
    plt.title("Distribution of Final Grades (G3)")
    plt.show()