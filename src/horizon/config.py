import os


# Missing or invalid configuration: a problem the user fixes, so callers report
# it as a message and exit code rather than a traceback.
class ConfigError(Exception):
    pass


# Secrets come from the environment, never from committed files. `not url` also
# rejects a set-but-empty value, such as `DATABASE_URL=` in .env.
def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ConfigError(
            "DATABASE_URL is not set; export it or run with uv run --env-file .env"
        )

    return url


# Returns (key_id, secret). Each key is checked separately so the error names
# the one that is missing.
def alpaca_keys() -> tuple[str, str]:
    alpaca_key = os.environ.get("ALPACA_API_KEY")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")

    if not alpaca_key:
        raise ConfigError(
            "ALPACA_API_KEY is not set; export it or run with uv run --env-file .env"
        )

    if not alpaca_secret:
        raise ConfigError(
            "ALPACA_SECRET_KEY is not set; export it or run with uv run --env-file .env"
        )

    return alpaca_key, alpaca_secret
