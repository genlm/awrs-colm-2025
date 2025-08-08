import pytest
from genlm.backend import load_model_by_name


@pytest.fixture(scope="session")
def llm():
    return load_model_by_name("meta-llama/Llama-3.2-1B-Instruct", backend="hf")
