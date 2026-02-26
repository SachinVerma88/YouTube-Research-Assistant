"""
URL configuration for youtube_bot project.
"""

from django.urls import path
from app.telegram_handler import webhook_view, health_check

urlpatterns = [
    path("webhook/", webhook_view, name="telegram_webhook"),
    path("health/", health_check, name="health_check"),
]
