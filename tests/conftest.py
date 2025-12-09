"""
Pytest configuration and fixtures for bca_survival tests.
"""

import tempfile

import matplotlib
import pytest

# Use a non-interactive backend for tests
# This must be done before importing pyplot
matplotlib.use("Agg")


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    import shutil

    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_clinical_df():
    """Create a sample clinical dataframe for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    return pd.DataFrame(
        {
            "patient_id": range(1, 21),
            "age": np.random.normal(65, 10, 20),
            "gender": np.random.choice(["M", "F"], 20),
            "diagnosis_date": ["15.01.2020"] * 20,
            "event_date": [f"{(i % 28) + 1:02d}.{(i % 12) + 1:02d}.2021" for i in range(20)],
            "event_status": np.random.binomial(1, 0.3, 20),
        }
    )


@pytest.fixture
def sample_bca_df():
    """Create a sample BCA measurements dataframe for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    return pd.DataFrame(
        {
            "id": range(1, 21),
            "l5::WL::imat::mean_ml": np.random.normal(10, 3, 20),
            "l5::WL::tat::mean_ml": np.random.normal(50, 15, 20),
            "l5::WL::muscle::mean_ml": np.random.normal(100, 25, 20),
        }
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "requires_pdftk: marks tests that require pdftk to be installed"
    )
    config.addinivalue_line(
        "markers", "requires_fpdf: marks tests that require fpdf to be installed"
    )
