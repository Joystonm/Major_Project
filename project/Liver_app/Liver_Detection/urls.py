# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.home, name='home'),
#      path('detection/', views.detection, name='detection'),
# ]

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('detection/', views.detect_tumor, name='detection'),
]

# Add this line to serve static files during development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
