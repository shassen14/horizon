# packages/ml_core/training/strategies.py

from abc import ABC, abstractmethod
import pandas as pd
import lightgbm as lgb
import inspect

from packages.contracts.blueprints import TrainingConfig


class TrainingStrategy(ABC):
    """
    Abstract Base Class for all model training procedures.
    """

    def __init__(self, training_config: TrainingConfig):
        self.config = training_config

    @abstractmethod
    def train(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame,
        y_val: pd.DataFrame,
    ):
        """
        Takes an un-trained model and training/validation data,
        and returns the trained model artifact.
        """
        pass


class SklearnTrainingStrategy(TrainingStrategy):
    """Handles training for any model with a scikit-learn compatible .fit() API."""

    def train(self, model, X_train, y_train, X_val, y_val):
        print(f"--- Using SklearnTrainingStrategy for {type(model).__name__} ---")

        fit_params = {}
        # Check if the model supports early stopping (like LightGBM/XGBoost)
        # We inspect the arguments of the 'fit' method for robustness
        try:
            fit_args = inspect.signature(model.fit).parameters
        except TypeError:  # Some objects might not have inspectable signatures
            fit_args = {}

        if "eval_set" in fit_args and self.config.early_stopping_rounds > 0:
            print(
                f"Configuring early stopping with {self.config.early_stopping_rounds} rounds."
            )
            fit_params["eval_set"] = [(X_val, y_val.values.ravel())]

            # eval_metric is often a param for the fit method itself
            if "eval_metric" in fit_args:
                fit_params["eval_metric"] = self.config.eval_metric

            # Callbacks are a more modern way to handle this, especially in LightGBM
            if "callbacks" in fit_args:
                fit_params["callbacks"] = [
                    lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)
                ]

        # Execute Training
        # Use .values.ravel() to convert DataFrame column to a 1D Numpy array,
        # which is the standard expected format and suppresses DataConversionWarning.
        model.fit(X_train, y_train.values.ravel(), **fit_params)

        return model


class PyTorchTrainingStrategy(TrainingStrategy):
    """Handles the training loop for a PyTorch nn.Module."""

    def train(self, model, X_train, y_train, X_val, y_val):
        print("--- Using PyTorchTrainingStrategy ---")
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # 1. Convert data to Tensors
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

        # 2. Create DataLoaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        # 3. Get Optimizer and Loss from Config
        optimizer_class = getattr(torch.optim, self.config.optimizer)
        optimizer = optimizer_class(model.parameters(), lr=self.config.learning_rate)

        loss_fn_class = getattr(torch.nn, self.config.loss_function)
        loss_fn = loss_fn_class()

        # 4. The Training Loop
        for epoch in range(self.config.epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                # Forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # (Add validation logic here if needed)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        return model
