from django.contrib import admin
from .models import Provider, ProviderAddress, ProviderTaxonomy, TaxonomyCode

@admin.register(TaxonomyCode)
class TaxonomyCodeAdmin(admin.ModelAdmin):
    list_display = ('code', 'classification', 'specialization')
    search_fields = ('code', 'classification')

@admin.register(Provider)
class ProviderAdmin(admin.ModelAdmin):
    list_display = ('npi', 'full_name', 'entity_type')
    search_fields = ('npi', 'first_name', 'last_name', 'organization_name')
    list_filter = ('entity_type',)

@admin.register(ProviderAddress)
class ProviderAddressAdmin(admin.ModelAdmin):
    list_display = ('provider', 'address_type', 'city', 'state')
    list_filter = ('address_type', 'state')

@admin.register(ProviderTaxonomy)
class ProviderTaxonomyAdmin(admin.ModelAdmin):
    list_display = ('provider', 'taxonomy_code', 'is_primary')
    list_filter = ('is_primary',)
