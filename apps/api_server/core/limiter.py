# apps/api_server/core/limiter.py

from slowapi import Limiter
from fastapi import Request


def get_real_ip(request: Request) -> str:
    # 1. Try Cloudflare Header (Standard for Tunnels)
    if request.headers.get("cf-connecting-ip"):
        return request.headers["cf-connecting-ip"]

    # 2. Try X-Forwarded-For (Standard Proxy)
    if request.headers.get("x-forwarded-for"):
        return request.headers["x-forwarded-for"].split(",")[0]

    # 3. Fallback to direct IP (Localhost dev)
    return request.client.host or "127.0.0.1"


# Create and export the singleton instance of the limiter.
# Any part of the app that imports this will get the exact same object.
limiter = Limiter(key_func=get_real_ip)
