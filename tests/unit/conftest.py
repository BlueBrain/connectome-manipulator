"""Testing helpers"""
import pytest


@pytest.fixture
def nodes():
    class FakeNode:
        config = {"morphologies_dir": "/foo/bar"}
        _population = None

    return (FakeNode(), FakeNode())
