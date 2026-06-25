import os, yaml

# Resolve project root relative to this test file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def test_compose_file_has_all_services():
    with open(os.path.join(PROJECT_ROOT, "docker-compose.yml")) as f:
        cfg = yaml.safe_load(f)
    services = set(cfg["services"].keys())
    # prepare-nn-data removed in phase-11 (NN grouping is internal to NNOrchestrator.train)
    required = {"graber","ohlc_gen","simulate","nn-train","simulate-nn","live","view-sim","view-full","view-live"}
    assert required == services


def test_no_secrets_in_compose():
    with open(os.path.join(PROJECT_ROOT, "docker-compose.yml")) as f:
        content = f.read()
    assert "BINANCE_API_KEY=" not in content
    assert "BINANCE_API_SECRET=" not in content


def test_live_env_has_paper_trade_flag():
    with open(os.path.join(PROJECT_ROOT, "configs/live.env")) as f:
        content = f.read()
    assert "IS_TRAIDER_TEST=1" in content


def test_vol_creation_script_exists_and_executable():
    path = os.path.join(PROJECT_ROOT, "scripts/vol_creation.sh")
    assert os.path.exists(path)
    assert os.access(path, os.X_OK)


def test_env_file_defaults_present():
    with open(os.path.join(PROJECT_ROOT, ".env")) as f:
        content = f.read()
    for var in ["FUNCTIONAL_ENV","TRAIN_ENV","VALIDATE_ENV","LONG_ENV","NN_TRAIN_ENV","LIVE_ENV"]:
        assert var in content
