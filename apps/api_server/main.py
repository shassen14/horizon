# apps/api_server/main.py

from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Absolute, clean imports
from packages.quant_lib.logging import LogManager
from apps.api_server.core.limiter import limiter
from apps.api_server.routers import public as public_router
from apps.api_server.routers import assets as assets_router


# Init Logger
log_manager = LogManager(service_name="api_server")
logger = log_manager.get_logger("main")

# Create App
app = FastAPI(title="Horizon API", version="1.0.0")

# Attach State & Handlers
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include Routers with a global prefix
app.include_router(public_router.router, prefix="/api/v1", tags=["Public Market Data"])
app.include_router(assets_router.router, prefix="/api/v1", tags=["Public Asset Data"])


@app.get("/", tags=["Health Check"], operation_id="health_check")
def read_root():
    """A simple health check endpoint."""
    logger.info("Health check endpoint was hit.")
    return {"status": "ok", "service": "Horizon API"}
