"""
Management command to set up the Telegram webhook.

Usage:
    python manage.py setup_webhook --url https://your-domain.ngrok.io/webhook/
    python manage.py setup_webhook --delete   # Remove webhook
"""

import asyncio
from django.core.management.base import BaseCommand
from django.conf import settings
from telegram import Bot


class Command(BaseCommand):
    help = "Set up or remove the Telegram bot webhook"

    def add_arguments(self, parser):
        parser.add_argument(
            "--url",
            type=str,
            help="Public webhook URL (e.g. https://abc123.ngrok.io/webhook/)",
        )
        parser.add_argument(
            "--delete",
            action="store_true",
            help="Delete the existing webhook",
        )

    def handle(self, *args, **options):
        token = settings.TELEGRAM_BOT_TOKEN
        if not token:
            self.stderr.write(self.style.ERROR(
                "TELEGRAM_BOT_TOKEN is not set in .env"
            ))
            return

        bot = Bot(token=token)

        if options["delete"]:
            asyncio.run(self._delete_webhook(bot))
        elif options.get("url"):
            asyncio.run(self._set_webhook(bot, options["url"]))
        else:
            # Use WEBHOOK_URL from settings
            url = settings.WEBHOOK_URL
            if not url:
                self.stderr.write(self.style.ERROR(
                    "Provide --url or set WEBHOOK_URL in .env"
                ))
                return
            asyncio.run(self._set_webhook(bot, url))

    async def _set_webhook(self, bot: Bot, url: str):
        await bot.set_webhook(url=url)
        info = await bot.get_webhook_info()
        self.stdout.write(self.style.SUCCESS(f"✅ Webhook set to: {info.url}"))
        self.stdout.write(f"   Pending updates: {info.pending_update_count}")

    async def _delete_webhook(self, bot: Bot):
        await bot.delete_webhook()
        self.stdout.write(self.style.SUCCESS("✅ Webhook deleted"))
