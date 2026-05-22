#!/usr/bin/env python3
"""
Iris Species Classification and Analysis System Pipeline
--------------------------------------------------------
This script implements a production-grade machine learning pipeline to analyze,
train, and evaluate multi-class classification models on the Iris dataset.

Features:
- Programmatic Command-Line Interface (CLI)
- Fully modular data preprocessing, training, evaluation, and inference steps
- Professional structured logging
- Dynamic visualization export
"""

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Global configurations
OUTPUT_DIR = "outputs"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def setup_logging(verbose: bool = False) -> None:
    """Configures the standard logging format and level."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Loads the raw Iris dataset from scikit-learn and parses it into a DataFrame.

    Returns:
        tuple[pd.DataFrame, list[str]]: A tuple containing the parsed DataFrame and target names.
    """
    logging.info("Loading raw Iris dataset from scikit-learn.")
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})
    return df, list(iris.target_names)

def run_eda(df: pd.DataFrame, target_names: list[str]) -> None:
    """
    Executes Exploratory Data Analysis (EDA) and exports visualization assets.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_names (list[str]): Names of the target classes.
    """
    logging.info(f"Initiating Exploratory Data Analysis. Shape: {df.shape}")
    logging.info(f"Class Distributions:\n{df['species'].value_counts().to_string()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Generate Pairplot
    logging.debug("Generating feature pairplot.")
    plt.figure(figsize=(10, 8))
    sns.pairplot(df.drop('target', axis=1), hue='species', palette='muted')
    plt.suptitle("Iris Feature Pairplot", y=1.02, fontsize=14)
    pairplot_path = os.path.join(OUTPUT_DIR, "iris_pairplot.png")
    plt.savefig(pairplot_path, bbox_inches='tight')
    plt.close()
    logging.info(f"EDA Pairplot exported to '{pairplot_path}'")

    # 2. Generate Correlation Heatmap
    logging.debug("Generating feature correlation heatmap.")
    plt.figure(figsize=(8, 6))
    features_only = df.drop(['target', 'species'], axis=1)
    sns.heatmap(features_only.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Iris Features")
    corr_path = os.path.join(OUTPUT_DIR, "iris_correlation.png")
    plt.savefig(corr_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Correlation heatmap exported to '{corr_path}'")

def train_and_evaluate(df: pd.DataFrame, target_names: list[str]) -> dict:
    """
    Splits data, trains models, evaluates metrics, and exports confusion matrices.

    Args:
        df (pd.DataFrame): Preprocessed dataframe.
        target_names (list[str]): Target class name mapping.

    Returns:
        dict: A dictionary of trained classifiers.
    """
    logging.info("Preparing data partition splits (80% Train / 20% Test).")
    features = df.columns[:4].tolist() # Sepal L/W, Petal L/W
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logging.info(f"Split metrics: Training size = {X_train.shape[0]}, Test size = {X_test.shape[0]}")

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
        "K-Nearest_Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Decision_Tree": DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for name, model in models.items():
        logging.info(f"Training classifier: {name}")
        model.fit(X_train, y_train)

        # Performance evaluation
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        logging.info(f"[{name}] Train Accuracy: {train_acc * 100:.2f}% | Test Accuracy: {test_acc * 100:.2f}%")

        # Log classification report details
        report_str = classification_report(y_test, test_pred, target_names=target_names)
        logging.info(f"[{name}] Classification Metrics:\n{report_str}")

        # Save Confusion Matrix visual
        cm = confusion_matrix(y_test, test_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.title(f'Confusion Matrix - {name.replace("_", " ")}')

        filename = os.path.join(OUTPUT_DIR, f"cm_{name.lower()}.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
        logging.debug(f"[{name}] Confusion matrix visual exported to '{filename}'")

    return models

def make_prediction(models: dict, input_data: list[float], target_names: list[str]) -> None:
    """
    Executes inference for a single input record across the trained models.

    Args:
        models (dict): Dictionary of trained models.
        input_data (list[float]): Sepal length, sepal width, petal length, petal width.
        target_names (list[str]): Target class labels.
    """
    # Define DataFrame with feature names to ensure compatibility and suppress warnings
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    new_flower = pd.DataFrame([input_data], columns=features)

    logging.info(f"Executing inference for input record: {input_data}")

    # We use KNN as the primary classifier representation for CLI feedback
    primary_model_name = "K-Nearest_Neighbors"
    if primary_model_name not in models:
        # Fallback to first available model
        primary_model_name = list(models.keys())[0]

    model = models[primary_model_name]
    pred_class_id = model.predict(new_flower)[0]
    pred_species = target_names[pred_class_id]

    logging.info(f"Predicted Species: {pred_species.upper()} (Class ID: {pred_class_id}) [via {primary_model_name.replace('_', ' ')}]")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(new_flower)[0]
        logging.info("Confidence Probabilities:")
        for idx, name in enumerate(target_names):
            logging.info(f"  - {name}: {probs[idx] * 100:.1f}%")

def main() -> None:
    """Application entrypoint."""
    parser = argparse.ArgumentParser(
        description="Iris Species Classification and Analysis Pipeline CLI Tool"
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Trigger full preprocessing, model training, benchmarking, and export visualization assets"
    )
    parser.add_argument(
        "--predict",
        type=float,
        nargs=4,
        metavar=('SL', 'SW', 'PL', 'PW'),
        help="Execute classification inference for a single flower by providing 4 space-separated measurements (in cm): SepalLength SepalWidth PetalLength PetalWidth"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable diagnostic verbose/debug logs"
    )

    args = parser.parse_args()

    # Setup logger
    setup_logging(args.verbose)

    # Default behavior: run train + predict default sample if no action arguments are provided
    if not (args.train or args.predict):
        logging.info("No action arguments provided. Defaulting to full system run (--train + sample predict).")
        args.train = True
        default_predict_sample = [5.1, 3.5, 1.4, 0.2]
    else:
        default_predict_sample = None

    try:
        # Load dataset
        df, target_names = load_data()

        trained_models = {}
        if args.train:
            # Run EDA
            run_eda(df, target_names)
            # Run Train & Eval
            trained_models = train_and_evaluate(df, target_names)
            logging.info("Pipeline training and evaluation tasks completed successfully.")

        if args.predict or default_predict_sample:
            # If we didn't train in this execution run, train briefly to get model variables for inference
            if not trained_models:
                trained_models = train_and_evaluate(df, target_names)

            prediction_input = args.predict if args.predict else default_predict_sample
            make_prediction(trained_models, prediction_input, target_names)

    except Exception as e:
        logging.exception(f"Pipeline execution halted due to unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
