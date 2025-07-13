import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference
from ml.data import process_data

@pytest.fixture
def df():
    """
    Fixture to load sample census data
    """
    return pd.read_csv("/Users/brookedixon/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/data/census.csv")

# Test 1: Model training returns a fitted model
def test_train_fitted_model():
    """
    Test that train_model returns a RandomForestClassifier on trained sample data.
    """
    X = np.array([[0, 1], [1, 1], [0, 0], [1,0]])
    y = np.array([0, 1, 0, 1])
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# Test 2: Process_data encodes labels correctly
def test_apply_labels():
    """
    Test labels are correctly binarized by process_data using LabelBinarizer
    """
    # Create a small sample DataFrame for testing
    df = pd.DataFrame({
    "feature_1": ["A", "B", "A", "B", "A", "B", "A", "B"],
    "feature_2": ["X", "Y", "X", "Z", "Z", "Y", "X", "X"],
    "salary": ["<=50K", ">50K", "<=50K", "<=50K", "<=50K", ">50K", "<=50K", ">50K"]
    })
    
    cat_features = ["feature_1", "feature_2"]
    
    # Run process_data with training=True so that it fits the LabelBinarizer
    X, y, encoder, lb = process_data(
        df,
        categorical_features = cat_features,
        label = "salary",
        training = True
        )
    
    # Check that y is correctly binarized
    assert set(y) == {0,1}, "Labels not properly binarized"
    assert y.shape[0] == df.shape[0], "Mismatch between labels and number of rows"
    assert lb.classes_.tolist() == ["<=50K", ">50K"], "LabelBinarizer"


# Test 3: Inference returns expected # of predictions
def test_inference_output():
    """
    # Test that inference output returns predictions of the correct shape
    """
    X = np.array([[0, 1], [1,0]])
    y = np.array([0, 1])
    model = train_model(X, y)
    preds = inference(model, X)
    
    assert preds.shape[0] == X.shape[0], "Number of predictions should match # of rows"
