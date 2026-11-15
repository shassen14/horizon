import pytest

from horizon.cli import main


def test_migrate_applies_then_is_up_to_date(
    empty_db: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DATABASE_URL", empty_db)

    assert main(["migrate"]) == 0
    assert capsys.readouterr().out.startswith("applied migrations:")

    # A second run finds nothing pending and still succeeds.
    assert main(["migrate"]) == 0
    assert capsys.readouterr().out == "already up to date\n"


@pytest.mark.parametrize("value", [None, ""])
def test_migrate_without_database_url_fails(
    value: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Unset and set-but-empty are both treated as missing.
    if value is None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
    else:
        monkeypatch.setenv("DATABASE_URL", value)

    assert main(["migrate"]) == 1

    # The error goes to stderr, so stdout stays clean for anything reading it.
    captured = capsys.readouterr()
    assert "DATABASE_URL" in captured.err
    assert captured.out == ""


def test_no_command_is_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    # argparse exits by itself on bad input, with the conventional usage code 2.
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2
    assert "usage: horizon" in capsys.readouterr().err
