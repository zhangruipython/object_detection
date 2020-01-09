from django.apps import AppConfig
from django.utils.module_loading import autodiscover_modules


class ObjectCountAppConfig(AppConfig):
    name = 'object_count_app'

    def ready(self):
        autodiscover_modules("views.py")
