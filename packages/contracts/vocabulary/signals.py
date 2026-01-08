from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


class Action(StrEnum):
    """Defines the desired action for a specific asset or the portfolio as a whole."""

    # Asset-level actions
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"
    INCREASE_LONG = "INCREASE_LONG"
    DECREASE_LONG = "DECREASE_LONG"

    # Portfolio-level actions (from the Risk Engine)
    HEDGE = "HEDGE"  # Add a negatively correlated asset (e.g., buy TLT)
    GO_CASH = "GO_CASH"  # Exit all risk assets
    REBALANCE = "REBALANCE"  # Adjust weights to match target, no new assets


class Urgency(StrEnum):
    """Defines how quickly an action should be taken."""

    LOW = "LOW"  # Can be done end-of-day or over several hours
    MEDIUM = "MEDIUM"  # Should be done within the hour
    HIGH = "HIGH"  # Execute immediately (e.g., market order for stop-loss)
