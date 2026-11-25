import pytest

import backend.app as app_module
from backend.memory.store import PatientStore


@pytest.fixture(autouse=True)
def private_store(tmp_path, monkeypatch):
    """Tests never touch the real patient database."""
    store = PatientStore(tmp_path / "private" / "test.db")
    monkeypatch.setattr(app_module, "store", store)
    return store
