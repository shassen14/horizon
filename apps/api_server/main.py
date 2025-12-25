# apps/api_server/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Absolute, clean imports
from packages.quant_lib.config import settings
from packages.quant_lib.logging import LogManager
from apps.api_server.core.limiter import limiter
from apps.api_server.routers import market as public_router
from apps.api_server.routers import assets as assets_router
from apps.api_server.routers import discovery as discovery_router
from apps.api_server.routers import system as system_router
from apps.api_server.routers import intelligence as intelligence_router


# Init Logger
log_manager = LogManager(service_name="api_server")
logger = log_manager.get_logger("main")

# Create App
app = FastAPI(
    title=settings.system.project_name,
    version=settings.system.version,
    debug=settings.system.debug,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.system.allowed_origins_list,  # List of allowed origins
    allow_credentials=True,  # Allow cookies/auth headers
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE)
    allow_headers=["*"],  # Allow all headers
)

# Attach State & Handlers
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include Routers with a global prefix
api_prefix = "/api/v1"

app.include_router(public_router.router, prefix=api_prefix)
app.include_router(assets_router.router, prefix=api_prefix)
app.include_router(discovery_router.router, prefix=api_prefix)
app.include_router(system_router.router, prefix=api_prefix)
app.include_router(intelligence_router.router, prefix=api_prefix)


@app.get("/", tags=["Health Check"], operation_id="health_check")
def read_root():
    """A simple health check endpoint."""
    logger.info("Health check endpoint was hit.")
    return {"status": "ok", "service": "Horizon API"}
