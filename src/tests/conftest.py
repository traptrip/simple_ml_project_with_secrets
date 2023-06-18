# see: https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
from pathlib import Path

import pytest

from src.utils import read_config
from src.net import Net

DEFAULT_CFG_PATH = Path(__file__).parent / "../../config.yml"


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow", action="store_true", default=False, help="skip slow tests"
    )
    parser.addoption(
        "--skip-integration", action="store_true", default=False, help="skip slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-slow"):
        skip_slow = pytest.mark.skip(
            reason="you need to remove --skip-slow option to run"
        )
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    if config.getoption("--skip-integration"):
        skip_slow = pytest.mark.skip(
            reason="you need to remove --skip-integration option to run"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_slow)


@pytest.fixture()
def config():
    cfg = read_config(DEFAULT_CFG_PATH)
    cfg.data.data_dir = "tests/tiny_dataset"
    cfg.train.net.num_classes = 2
    cfg.train.loss.num_classes = 2
    cfg.train.batch_size = 2
    cfg.infer.batch_size = 2
    return cfg


@pytest.fixture()
def model(config):
    net = Net(config.train)
    net.to(config.infer.device)
    net.eval()
    return net
