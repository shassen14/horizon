# packages/ml_core/common/tracker.py

import mlflow
import pandas as pd
import pkg_resources
import polars as pl
from typing import Dict, Any, List
from pathlib import Path
from mlflow.models import ModelSignature, infer_signature
from packages.quant_lib.config import settings
from packages.quant_lib.helpers.structs import flatten_dict


class ExperimentTracker:
    """
    Context Manager wrapper around MLflow.
    Handles connection, zombie killing, parameter flattening, and safe logging.
    """

    def __init__(self, experiment_name: str, run_name: str = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None
        self.run_id = None

    def __enter__(self):
        # 1. Setup Connection with Fallback
        target_uri = settings.mlflow.tracking_uri

        try:
            mlflow.set_tracking_uri(target_uri)
            # Simple check to see if server is reachable
            mlflow.get_experiment_by_name(self.experiment_name)
        except Exception as e:
            print(f"⚠️  MLflow Server ({target_uri}) unreachable: {e}")
            print("⚠️  Falling back to local 'file:./mlruns' to save progress.")
            mlflow.set_tracking_uri("file:./mlruns")

        # 2. Setup Experiment
        try:
            # Check if experiment exists
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            # Scenario 1: Experiment does NOT exist
            if experiment is None:
                print(f"Creating new experiment '{self.experiment_name}'...")
                mlflow.create_experiment(
                    name=self.experiment_name, artifact_location="mlflow-artifacts:/"
                )

            # Scenario 2: Experiment EXISTS, but has the WRONG artifact location
            elif experiment.artifact_location != "mlflow-artifacts:/":
                print(
                    f"⚠️  Experiment '{self.experiment_name}' has incorrect artifact location: {experiment.artifact_location}"
                )
                print(
                    f"⚠️  This can happen if it was created by an older version of the code."
                )
                # For now, we will NOT proceed to avoid writing to the wrong place.
                # In a fully automated system, you might delete and recreate it here.
                raise SystemExit(
                    f"Exiting due to misconfigured experiment '{self.experiment_name}'."
                )

            # Activate it
            mlflow.set_experiment(experiment_name=self.experiment_name)

        except Exception as e:
            print(f"⚠️ Experiment setup failed: {e}")
            # If server fails, we might crash here or fall back,
            # but let's allow flow to continue to see specific errors
            pass

        # 3. Zombie Killer
        if mlflow.active_run():
            print(f"🧹 Killing zombie run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()

        # 4. Start Run
        self.run = mlflow.start_run(run_name=self.run_name)
        self.run_id = self.run.info.run_id
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
        if exc_type:
            print(f"❌ Run {self.run_id} failed with exception.")
        else:
            print(f"✅ Run {self.run_id} completed successfully.")

    def log_params(self, params: Dict[str, Any]):
        """automatically flattens nested dicts before logging"""
        flat_params = flatten_dict(params)
        mlflow.log_params(flat_params)

    def log_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def set_tags(self, tags: Dict[str, str]):
        mlflow.set_tags(tags)

    def get_run_id(self) -> str:
        return self.run_id

    def log_dataset(self, df: pl.DataFrame, name: str, cache_key: str):
        """
        Logs the dataset METADATA (Hash, Schema, Stats) to MLflow.
        Does NOT upload the physical file to save space (rely on Config+DB to reproduce).
        """
        # 1. Log the Cache Key as a Tag
        # This tells us exactly which 'version' of the data config was used
        mlflow.set_tag("dataset_cache_key", cache_key)

        # 2. Log Input (The Fingerprint)
        # We convert to Pandas because MLflow's native Polars support
        # is still evolving and Pandas is the stable standard for signatures.
        try:

            # Convert to pandas (zero-copy if possible) for metadata extraction
            pdf = df.to_pandas()

            #  Cast Ints to Floats for MLflow Schema Safet
            # This prevents the "Integer columns cannot represent missing values" warning.
            # We do this only on the copy used for logging.
            int_cols = pdf.select_dtypes(
                include=["int64", "int32", "int16", "int8"]
            ).columns
            if len(int_cols) > 0:
                # Convert to float64 to allow NaNs in MLflow's schema definition
                pdf[int_cols] = pdf[int_cols].astype("float64")

            # Create MLflow Dataset object
            dataset = mlflow.data.from_pandas(pdf, name=name, digest=None)
            # Create MLflow Dataset object
            # This calculates a hash of the DATA CONTENT automatically.
            dataset = mlflow.data.from_pandas(
                pdf, name=name, digest=None  # Let MLflow calculate the digest/hash
            )

            # Log the inputs
            mlflow.log_input(dataset, context="training")

        except Exception as e:
            print(f"⚠️ Could not log dataset metadata: {e}")

    def log_environment(self, dependencies: List[str]):
        """
        Logs a minimal requirements.txt file based on the config.
        """
        print(f"Logging minimal environment for dependencies: {dependencies}")

        try:
            # Get versions for specified packages
            installed = {pkg.key for pkg in pkg_resources.working_set}
            reqs = []
            for dep in dependencies:
                if dep in installed:
                    version = pkg_resources.get_distribution(dep).version
                    reqs.append(f"{dep}=={version}")
                else:
                    reqs.append(dep)  # Add without version if not found

            reqs_text = "\n".join(reqs)

            # Use mlflow.log_text to avoid creating local files
            mlflow.log_text(reqs_text, "requirements.txt")

        except Exception as e:
            print(f"⚠️ Could not log requirements: {e}")

        # 2. Log uv.lock (The Source of Truth for uv)
        # Assuming project root is 3 levels up from this file
        try:
            project_root = Path(__file__).resolve().parents[3]
            lock_file = project_root / "uv.lock"

            if lock_file.exists():
                self.log_artifact(str(lock_file))
        except Exception as e:
            print(f"⚠️ Could not log uv.lock: {e}")

    def log_model_signature(
        self, X_sample: pd.DataFrame, y_sample: pd.DataFrame
    ) -> ModelSignature | None:
        """
        Infers the model's input/output schema and logs it as a JSON artifact.
        Useful for validating inputs during inference later.
        """
        try:
            # 1. Identify integer columns in the input sample
            int_cols = X_sample.select_dtypes(include=["int", "int32", "int64"]).columns

            if len(int_cols) > 0:
                print(
                    f"INFO: Casting integer columns for signature to avoid NaN issues: {list(int_cols)}"
                )
                # Create a copy and cast to float to make the signature robust
                X_sample_safe = X_sample.copy()
                X_sample_safe[int_cols] = X_sample_safe[int_cols].astype("float64")
            else:
                X_sample_safe = X_sample

            # 1. Infer the standard MLflow signature
            # This inspects column types and shapes automatically
            signature = infer_signature(X_sample_safe, y_sample)

            # 2. Convert to Dictionary
            # This creates a standard JSON representation of inputs/outputs
            sig_dict = signature.to_dict()

            # 3. Log directly to MLflow
            # mlflow.log_dict automatically creates the file and uploads it
            mlflow.log_dict(sig_dict, "model_signature.json")

            return signature

        except Exception as e:
            # Don't crash training if schema inference fails (e.g. complex types)
            print(f"⚠️ Could not infer/log model signature: {e}")
            return None

    def log_data_profile(self, df: pl.DataFrame):
        """Logs basic statistics about the training data."""
        try:
            description = df.describe().to_pandas()
            description.to_html("data_profile.html")
            self.log_artifact("data_profile.html")
            Path("data_profile.html").unlink()
        except Exception as e:
            print(f"⚠️ Could not log profile: {e}")
