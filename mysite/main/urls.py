from django.urls import path, include

from . import views
from django.contrib import admin
urlpatterns = [
    path("home/", views.home, name='home'),
    path("idea/", views.idea, name='idea'),
    path("team/", views.team, name='team'),
    path("test/", views.test, name='test'),
    path('admin/', admin.site.urls),
    path("", views.home, name='home'),
    # path('', include("django.contrib.auth.urls")),

]