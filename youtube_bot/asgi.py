"""
ASGI config for youtube_bot project.
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "youtube_bot.settings")
application = get_asgi_application()
