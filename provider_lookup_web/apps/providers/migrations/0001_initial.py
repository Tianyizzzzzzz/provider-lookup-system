# Generated by Django 5.2.4 on 2025-07-17 02:09

import django.contrib.postgres.indexes
import django.contrib.postgres.search
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TaxonomyCode',
            fields=[
                ('code', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('classification', models.CharField(max_length=200)),
                ('specialization', models.CharField(blank=True, max_length=200)),
                ('definition', models.TextField(blank=True)),
            ],
            options={
                'verbose_name': 'Taxonomy Code',
                'verbose_name_plural': 'Taxonomy Codes',
                'db_table': 'taxonomy_codes',
            },
        ),
        migrations.CreateModel(
            name='Provider',
            fields=[
                ('npi', models.CharField(max_length=10, primary_key=True, serialize=False)),
                ('entity_type', models.CharField(choices=[('Individual', 'Individual'), ('Organization', 'Organization')], max_length=12)),
                ('organization_name', models.CharField(blank=True, max_length=200)),
                ('last_name', models.CharField(blank=True, max_length=100)),
                ('first_name', models.CharField(blank=True, max_length=100)),
                ('middle_name', models.CharField(blank=True, max_length=100)),
                ('name_prefix', models.CharField(blank=True, max_length=10)),
                ('name_suffix', models.CharField(blank=True, max_length=10)),
                ('credential', models.CharField(blank=True, max_length=100)),
                ('phone', models.CharField(blank=True, max_length=20)),
                ('fax', models.CharField(blank=True, max_length=20)),
                ('enumeration_date', models.DateField(blank=True, null=True)),
                ('last_update_date', models.DateField(blank=True, null=True)),
                ('deactivation_date', models.DateField(blank=True, null=True)),
                ('search_vector', django.contrib.postgres.search.SearchVectorField(blank=True, null=True)),
            ],
            options={
                'db_table': 'providers',
                'indexes': [models.Index(fields=['last_name', 'first_name'], name='providers_last_na_a63836_idx'), models.Index(fields=['organization_name'], name='providers_organiz_80f83f_idx'), django.contrib.postgres.indexes.GinIndex(fields=['search_vector'], name='providers_search__732c9a_gin')],
            },
        ),
        migrations.CreateModel(
            name='ProviderAddress',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('address_type', models.CharField(choices=[('location', 'Practice Location'), ('mailing', 'Mailing Address')], max_length=10)),
                ('address_line_1', models.CharField(max_length=200)),
                ('address_line_2', models.CharField(blank=True, max_length=200)),
                ('city', models.CharField(max_length=100)),
                ('state', models.CharField(max_length=2)),
                ('postal_code', models.CharField(max_length=10)),
                ('country_code', models.CharField(default='US', max_length=2)),
                ('provider', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='addresses', to='providers.provider')),
            ],
            options={
                'db_table': 'provider_addresses',
                'indexes': [models.Index(fields=['city', 'state'], name='provider_ad_city_f036ce_idx'), models.Index(fields=['postal_code'], name='provider_ad_postal__549d07_idx'), models.Index(fields=['provider', 'address_type'], name='provider_ad_provide_b9b8f0_idx')],
            },
        ),
        migrations.CreateModel(
            name='ProviderTaxonomy',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('is_primary', models.BooleanField(default=False)),
                ('license_number', models.CharField(blank=True, max_length=50)),
                ('license_number_state', models.CharField(blank=True, max_length=2)),
                ('provider', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='taxonomies', to='providers.provider')),
                ('taxonomy_code', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='providers.taxonomycode')),
            ],
            options={
                'db_table': 'provider_taxonomies',
                'indexes': [models.Index(fields=['provider', 'is_primary'], name='provider_ta_provide_2e0fe0_idx'), models.Index(fields=['taxonomy_code'], name='provider_ta_taxonom_c2b4b3_idx')],
                'unique_together': {('provider', 'taxonomy_code')},
            },
        ),
    ]
