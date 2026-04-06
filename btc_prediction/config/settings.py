"""Comprehensive configuration settings for the BTC prediction system.

All configuration models are frozen (immutable) and can be overridden via
environment variables with the ``BTC_PRED_`` prefix.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel, frozen=True):
    """Neural network architecture configuration."""

    temporal_dim: int = Field(default=512, description="Temporal embedding dimension")
    orderbook_dim: int = Field(default=256, description="Order-book embedding dimension")
    onchain_dim: int = Field(default=256, description="On-chain embedding dimension")
    cross_asset_dim: int = Field(default=256, description="Cross-asset embedding dimension")
    sentiment_dim: int = Field(default=256, description="Sentiment embedding dimension")
    fused_dim: int = Field(default=1536, description="Fused representation dimension")
    num_regimes: int = Field(default=5, description="Number of market regimes")
    num_quantiles: int = Field(default=9, description="Number of prediction quantiles")
    dropout: float = Field(default=0.2, description="Dropout probability")
    sequence_length: int = Field(default=100, description="Input sequence length")
    num_heads: int = Field(default=8, description="Number of attention heads")
    num_transformer_layers: int = Field(default=4, description="Number of transformer layers")


class DataConfig(BaseModel, frozen=True):
    """Data ingestion and processing configuration."""

    exchanges: list[str] = Field(
        default=["binance", "coinbase", "kraken", "bybit"],
        description="Exchanges to ingest data from",
    )
    timeframes: list[str] = Field(
        default=["1m", "5m", "15m", "1h", "4h"],
        description="Candlestick timeframes to collect",
    )
    orderbook_levels: int = Field(default=10, description="Order-book depth levels")
    lookback_window: int = Field(default=500, description="Historical lookback in bars")
    batch_size: int = Field(default=64, description="Training batch size")


class TrainingConfig(BaseModel, frozen=True):
    """Training hyper-parameters."""

    learning_rate: float = Field(default=1e-4, description="Optimizer learning rate")
    weight_decay: float = Field(default=1e-5, description="L2 regularisation weight")
    max_epochs: int = Field(default=100, description="Maximum training epochs")
    patience: int = Field(default=20, description="Early-stopping patience")
    grad_clip: float = Field(default=1.0, description="Gradient clipping norm")
    loss_weights: dict[str, float] = Field(
        default={
            "direction": 1.0,
            "magnitude": 0.5,
            "volatility": 0.3,
            "regime": 0.2,
        },
        description="Multi-task loss component weights",
    )


class InferenceConfig(BaseModel, frozen=True):
    """Real-time inference configuration."""

    latency_target_ms: int = Field(default=100, description="Target inference latency in ms")
    min_confidence: float = Field(default=0.6, description="Minimum prediction confidence")
    ensemble_size: int = Field(default=10, description="Number of ensemble members")


class RiskConfig(BaseModel, frozen=True):
    """Risk management parameters."""

    max_position_pct: float = Field(default=0.25, description="Maximum position size as % of NAV")
    base_stop_loss_pct: float = Field(default=0.02, description="Base stop-loss percentage")
    max_drawdown_pct: float = Field(default=0.20, description="Maximum allowable drawdown")


class FeatureConfig(BaseModel, frozen=True):
    """Feature engineering configuration."""

    technical_indicators: list[str] = Field(
        default=[
            "rsi",
            "macd",
            "bollinger_bands",
            "atr",
            "obv",
            "vwap",
            "ema_12",
            "ema_26",
            "stochastic",
            "adx",
        ],
        description="Technical indicators to compute",
    )
    onchain_metrics: list[str] = Field(
        default=[
            "active_addresses",
            "transaction_volume",
            "exchange_inflow",
            "exchange_outflow",
            "miner_revenue",
            "hash_rate",
            "mempool_size",
            "nvt_ratio",
        ],
        description="On-chain metrics to incorporate",
    )


class Settings(BaseSettings, frozen=True):
    """Top-level application settings.

    All fields can be overridden via environment variables prefixed with
    ``BTC_PRED_`` (e.g. ``BTC_PRED_DEBUG=true``).
    """

    model_config = SettingsConfigDict(env_prefix="BTC_PRED_", extra="ignore")

    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)


def get_settings() -> Settings:
    """Create and return the application settings instance."""
    return Settings()
