from django.urls import include, path
from django.contrib import admin
from django.conf.urls.static import static
from django.conf import settings
from unwrap.views import show, upload, download, mobile
urlpatterns = [
    path('admin/', admin.site.urls),
    path('upload/', upload),
    path('mobile/', mobile),
    path('download/', download),
    path('show/', show),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 