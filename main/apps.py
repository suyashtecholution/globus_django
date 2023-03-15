from django.apps import AppConfig
import os
from .Arducam import *


class MainConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "main"


class MyAppConfig(AppConfig):
    name = 'myapp'
    verbose_name = "My Application"

    def ready(self):
        camera = ArducamCamera()
        try:

            config_file = os.path.join(os.path.dirname(__file__), 'ardu.cfg')

            if not camera.openCamera(config_file):
                raise RuntimeError("Failed to open camera.")

            camera.start()
        except Exception as e:
            camera.stop()
            camera.closeCamera()

