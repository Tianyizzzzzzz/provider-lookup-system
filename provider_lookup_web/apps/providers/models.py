from django.db import models
from django.contrib.postgres.search import SearchVectorField
from django.contrib.postgres.indexes import GinIndex

class TaxonomyCode(models.Model):
    code = models.CharField(max_length=10, primary_key=True)
    classification = models.CharField(max_length=200)
    specialization = models.CharField(max_length=200, blank=True)
    definition = models.TextField(blank=True)

    class Meta:
        db_table = 'taxonomy_codes'
        verbose_name = 'Taxonomy Code'
        verbose_name_plural = 'Taxonomy Codes'

    def __str__(self):
        return f"{self.code} - {self.classification}"

class Provider(models.Model):
    ENTITY_TYPES = [
        ('Individual', 'Individual'),
        ('Organization', 'Organization'),
    ]

    npi = models.CharField(max_length=10, primary_key=True)
    entity_type = models.CharField(max_length=12, choices=ENTITY_TYPES)
    organization_name = models.CharField(max_length=200, blank=True)
    last_name = models.CharField(max_length=100, blank=True)
    first_name = models.CharField(max_length=100, blank=True)
    middle_name = models.CharField(max_length=100, blank=True)
    name_prefix = models.CharField(max_length=10, blank=True)
    name_suffix = models.CharField(max_length=10, blank=True)
    credential = models.CharField(max_length=100, blank=True)
    phone = models.CharField(max_length=20, blank=True)
    fax = models.CharField(max_length=20, blank=True)
    enumeration_date = models.DateField(null=True, blank=True)
    last_update_date = models.DateField(null=True, blank=True)
    deactivation_date = models.DateField(null=True, blank=True)
    search_vector = SearchVectorField(null=True, blank=True)

    class Meta:
        db_table = 'providers'
        indexes = [
            models.Index(fields=['last_name', 'first_name']),
            models.Index(fields=['organization_name']),
            GinIndex(fields=['search_vector']),
        ]

    @property
    def full_name(self):
        if self.entity_type == 'Individual':
            parts = [self.name_prefix, self.first_name, self.middle_name, 
                    self.last_name, self.name_suffix]
            return ' '.join(filter(None, parts))
        return self.organization_name

    def __str__(self):
        return f"{self.npi} - {self.full_name}"

class ProviderAddress(models.Model):
    ADDRESS_TYPES = [
        ('location', 'Practice Location'),
        ('mailing', 'Mailing Address'),
    ]

    provider = models.ForeignKey(Provider, on_delete=models.CASCADE, related_name='addresses')
    address_type = models.CharField(max_length=10, choices=ADDRESS_TYPES)
    address_line_1 = models.CharField(max_length=200)
    address_line_2 = models.CharField(max_length=200, blank=True)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=2)
    postal_code = models.CharField(max_length=10)
    country_code = models.CharField(max_length=2, default='US')

    class Meta:
        db_table = 'provider_addresses'
        indexes = [
            models.Index(fields=['city', 'state']),
            models.Index(fields=['postal_code']),
            models.Index(fields=['provider', 'address_type']),
        ]

    def __str__(self):
        return f"{self.provider.npi} - {self.city}, {self.state}"

class ProviderTaxonomy(models.Model):
    provider = models.ForeignKey(Provider, on_delete=models.CASCADE, related_name='taxonomies')
    taxonomy_code = models.ForeignKey(TaxonomyCode, on_delete=models.CASCADE)
    is_primary = models.BooleanField(default=False)
    license_number = models.CharField(max_length=50, blank=True)
    license_number_state = models.CharField(max_length=2, blank=True)

    class Meta:
        db_table = 'provider_taxonomies'
        unique_together = [['provider', 'taxonomy_code']]
        indexes = [
            models.Index(fields=['provider', 'is_primary']),
            models.Index(fields=['taxonomy_code']),
        ]

    def __str__(self):
        return f"{self.provider.npi} - {self.taxonomy_code.code}"
