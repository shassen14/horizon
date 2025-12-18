from pydantic import Field, PostgresDsn, computed_field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    user: str = Field(validation_alias="POSTGRES_USER", default="user")
    password: str = Field(validation_alias="POSTGRES_PASSWORD", default="password")
    host: str = Field(validation_alias="POSTGRES_HOST", default="localhost")
    port: int = Field(validation_alias="POSTGRES_PORT", default=5432)
    name: str = Field(validation_alias="POSTGRES_DB", default="horizon_db")

    @computed_field
    @property
    def URL(self) -> str:
        """Constructs the async SQLAlchemy connection string."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                path=self.name,
            )
        )
