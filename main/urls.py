"""globus_main URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from . import views
from django.views.decorators.csrf import csrf_exempt

from django.urls import path

urlpatterns = [
    path('take_image',views.capture_image,name='take_image'),
    path('close_camera',views.stop_django_server,name='close_camera'),
    path('start_camera',views.start_camera,name='start_camera'),
    path('inference_3_screw',views.inference_3_screw,name='inference_3_screw'),
    path('temp',views.temp_crop,name='temp'),
    path("manual_inference/from/<str:from>/to/<str:to>",views.manual_inference,name="manual_inference"),
    path('request_visualizer',csrf_exempt(views.request_visualizer),name='request_visualizer'),
    path('calib',views.calib,name='calib'),
]