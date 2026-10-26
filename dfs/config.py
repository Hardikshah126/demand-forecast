"""Central configuration. Secrets are read from environment variables / .env only."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# --- Project constants (kept in code so results are reproducible) ---------------------------
STORES = ("CA_1", "TX_1", "WI_1")
N_ITEMS = 100
HORIZON = 28
N_FOLDS = 3
SEED = 42
M5_LAST_DAY = 1913  # sales_train_validation.csv ends at d_1913
MODELS = ("naive", "seasonal_naive", "moving_average_28", "xgboost")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    database_url: str = Field(default="", repr=False)
    database_url_readonly: str = Field(default="", repr=False)
    readonly_db_user: str = "dfs_readonly"
    readonly_db_password: str = Field(default="", repr=False)
    allowed_origins: str = "http://localhost:3000"
    raw_data_dir: Path = ROOT / "data" / "raw"
    artifacts_dir: Path = ROOT / "artifacts"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


def get_settings() -> Settings:
    return Settings()
