from pydantic import Field, PostgresDsn, computed_field
from .base import EnvConfig


class MLflowConfig(EnvConfig):
    # Tracking Server Connection (For Clients)
    tracking_uri: str = Field(validation_alias="MLFLOW_TRACKING_URI")

    # Backend Database (For Server/Provisioning)
    db_user: str = Field(validation_alias="MLFLOW_DB_USER", default="mlflow")
    db_password: str = Field(validation_alias="MLFLOW_DB_PASSWORD", default="mlflow")
    db_name: str = Field(validation_alias="MLFLOW_DB_NAME", default="mlflow_db")

    # We reuse the main DB host/port since they live on the same storage
    # but we need to know them to build the connection string for provisioning
    db_host: str = Field(validation_alias="POSTGRES_HOST")
    db_port: int = Field(validation_alias="POSTGRES_PORT")

    @computed_field
    @property
    def DB_URL(self) -> str:
        """Synchronous connection string for Provisioning scripts."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+psycopg2",
                username=self.db_user,
                password=self.db_password,
                host=self.db_host,
                port=self.db_port,
                path=self.db_name,
            )
        )
