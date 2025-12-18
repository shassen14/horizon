# apps/api_server/core/limiter.py

from slowapi import Limiter
from slowapi.util import get_remote_address

# Define the key function once.
# This function tells the limiter how to identify a "user" (by their IP address).
limiter_key_func = get_remote_address

# Create and export the singleton instance of the limiter.
# Any part of the app that imports this will get the exact same object.
limiter = Limiter(key_func=limiter_key_func)
