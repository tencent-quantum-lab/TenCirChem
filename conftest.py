import pytest

from tencirchem import set_backend


@pytest.fixture
def reset_backend():
    yield
    set_backend("numpy")
