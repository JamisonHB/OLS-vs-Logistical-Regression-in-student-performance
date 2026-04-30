from preprocessing import load_data, preprocess_features, create_pass_fail, split_data, standardize
from sklearn.model_selection import train_test_split
from visualization import *
from ols_model import run_ols
from logistic_model import run_logistic

def main():
    # Load Data
    df = load_data()

    # Visuals
    plot_studytime_vs_grade(df)
    plot_absences(df)
    plot_correlation(df)
    plot_pairplot(df)
    plot_grade_distribution(df)

    # Preprocessing
    X, y = preprocess_features(df)

    # --- OLS Regression ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    run_ols(X_train, y_train, X_test, y_test)

    # --- Logistic Regression ---
    y_binary = create_pass_fail(y)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y_binary)
    X_train, X_val, X_test = standardize(X_train, X_val, X_test)
    
    run_logistic(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    main()