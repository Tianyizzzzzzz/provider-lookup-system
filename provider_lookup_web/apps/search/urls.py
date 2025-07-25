# apps/search/urls.py
from django.urls import path
from . import views

app_name = 'search'

urlpatterns = [
    path('', views.ProviderSearchView.as_view(), name='search'),
    path('provider/<str:npi>/', views.provider_detail, name='provider_detail'),
]