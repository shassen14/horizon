import os


class ConfigError(Exception):
    pass


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ConfigError(
            "DATABASE_URL is not set; export it or run with uv run --env-file .env"
        )

    return url
