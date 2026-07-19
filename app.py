from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt


def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    """
    Deserialize and load the regression model and use it to predict on user provided data.
    """

    model = load(filename)
    y = model.predict(X)

    return y


def create_streamlit_app():
    """
    Creates a Streamlit web application for making predictions with a simple regression model.
    """

    # Streamlit app title
    st.title("Linear Regression Predictor")

    # User input for new prediction using a slider
    input_feature = st.slider(
        "Select feature value",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
        step=0.1
    )

    # Button to make a prediction
    if st.button("Predict value"):

        # Call load_and_predict
        prediction = load_and_predict([[input_feature]])

        # Display prediction
        st.write(f"Predicted value: {prediction[0]:.2f}")

        # Visualize difference
        visualize_difference(input_feature, prediction[0])


def visualize_difference(input_feature: float, prediction: ArrayLike):
    """
    Visualize the difference between actual and predicted values.
    """

    # Load datasets
    X_filename = "X.joblib"
    y_filename = "y.joblib"

    X = load(X_filename)
    y = load(y_filename)

    actual_target = y[_index_of_closest(X, input_feature)]

    # Calculate difference
    difference = actual_target - prediction

    # Visualization
    fig = plt.figure(figsize=(6, 4))

    # Entire dataset
    plt.scatter(X, y, color="gray", label="Dataset")

    # Actual value
    plt.scatter(input_feature, actual_target, color="blue", s=100, label="Actual")

    # Predicted value
    plt.scatter(input_feature, prediction, color="red", s=100, label="Predicted")

    # Legend
    plt.legend()

    # Title
    plt.title("Actual vs Predicted Value")

    # Axis labels
    plt.xlabel("Feature")
    plt.ylabel("Target")

    # Grid
    plt.grid(True)

    # Difference line
    plt.plot(
        [input_feature, input_feature],
        [actual_target, prediction],
        "k--"
    )

    # Annotation
    plt.annotate(
        f"Diff: {difference:.2f}",
        xy=(input_feature, (actual_target + prediction) / 2),
        xytext=(10, 10),
        textcoords="offset points"
    )

    st.pyplot(fig)


# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()