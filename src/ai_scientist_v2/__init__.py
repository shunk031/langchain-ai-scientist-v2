from .config import settings

# Note: The `settings` read from the .env file will also update the OS environment variables. Therefore, always import and use this top-level settings.

__all__ = [
    "settings",
]
