import sys
sys.path.append(".")
import pytest
import src
from src.eval import load_data, Evaluation

@pytest.fixture(scope='module')
def get_data():
    return load_data()

@pytest.fixture(scope='module')
def get_instance():
    return Evaluation()

def test_load_data():
    """
    if load_data successfully using the correct dimension
    """
    load_data()

def test_f1(get_instance):
    tester = get_instance
    tester.cal_f1()
