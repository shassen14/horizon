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
