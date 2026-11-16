import pytest

from horizon.config import ConfigError, alpaca_keys


# Sets both keys, then lets each test remove or blank one. Starting from known
# values matters: the developer's shell may already export real keys, and the
# tests must not depend on that.
@pytest.fixture
def both_keys(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    monkeypatch.setenv("ALPACA_API_KEY", "test-key-id")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    return monkeypatch


def test_alpaca_keys_returns_both(both_keys: pytest.MonkeyPatch) -> None:
    assert alpaca_keys() == ("test-key-id", "test-secret")


# Each case should raise with a message naming the variable to fix.
@pytest.mark.parametrize("name", ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"])
def test_alpaca_keys_missing_raises(both_keys: pytest.MonkeyPatch, name: str) -> None:
    both_keys.delenv(name)

    with pytest.raises(ConfigError, match=name):
        alpaca_keys()


# `ALPACA_API_KEY=` in .env sets the variable to an empty string, which must
# fail like a missing one rather than send an empty header to Alpaca.
@pytest.mark.parametrize("name", ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"])
def test_alpaca_keys_empty_raises(both_keys: pytest.MonkeyPatch, name: str) -> None:
    both_keys.setenv(name, "")

    with pytest.raises(ConfigError, match=name):
        alpaca_keys()
