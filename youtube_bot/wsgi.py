"""
WSGI config for youtube_bot project.
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "youtube_bot.settings")
application = get_wsgi_application()
