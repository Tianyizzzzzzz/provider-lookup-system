# apps/search/views.py (fixed counting issue)
from django.shortcuts import render
from django.db.models import Q
from django.views.generic import ListView
from django.http import HttpResponse
import csv
from apps.providers.models import Provider


class ProviderSearchView(ListView):
    model = Provider
    template_name = 'search/search.html'
    context_object_name = 'results'
    paginate_by = 50  # Show 50 providers per page

    def get_queryset(self):
        # Get search parameters
        query = self.request.GET.get('query', '').strip()
        search_type = self.request.GET.get('search_type', 'all')
        city = self.request.GET.get('city', '').strip()
        state = self.request.GET.get('state', '').strip()

        if not any([query, city, state]):
            return Provider.objects.none()

        # Base queryset with optimized prefetching
        queryset = Provider.objects.select_related().prefetch_related(
            'addresses', 'taxonomies__taxonomy_code'
        )

        # Apply search filters based on search type
        if query:
            if search_type == 'npi':
                # NPI exact search
                if query.isdigit() and len(query) == 10:
                    queryset = queryset.filter(npi=query)
                else:
                    queryset = Provider.objects.none()

            elif search_type == 'name':
                # Provider name fuzzy search
                queryset = queryset.filter(
                    Q(first_name__icontains=query) |
                    Q(last_name__icontains=query) |
                    Q(organization_name__icontains=query)
                )

            else:  # search_type == 'all'
                # All fields search
                if query.isdigit() and len(query) == 10:
                    # If 10-digit number, prioritize NPI search
                    queryset = queryset.filter(npi=query)
                else:
                    # Otherwise search in all name fields
                    queryset = queryset.filter(
                        Q(first_name__icontains=query) |
                        Q(last_name__icontains=query) |
                        Q(organization_name__icontains=query) |
                        Q(npi__icontains=query)
                    )

        # Location search
        if city or state:
            address_filter = Q()
            if city:
                address_filter &= Q(addresses__city__icontains=city)
            if state:
                address_filter &= Q(addresses__state__iexact=state)

            queryset = queryset.filter(address_filter).distinct()

        # Add sorting
        return queryset.order_by('last_name', 'first_name', 'organization_name', 'npi')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Search parameters
        context['query'] = self.request.GET.get('query', '')
        context['search_type'] = self.request.GET.get('search_type', 'all')
        context['city'] = self.request.GET.get('city', '')
        context['state'] = self.request.GET.get('state', '')

        # Search statistics - fix counting issue
        if any([context['query'], context['city'], context['state']]):
            # Get count from complete queryset
            total_queryset = self.get_queryset()
            context['total_count'] = total_queryset.count()
            context['showing_limited'] = context['total_count'] > 50

            # Ensure results also have correct count
            if hasattr(context, 'object_list') and context['object_list']:
                context['results_count'] = len(context['object_list'])
            else:
                context['results_count'] = 0
        else:
            context['total_count'] = 0
            context['results_count'] = 0
            context['showing_limited'] = False

        # US States list
        context['states'] = [
            ('', 'Select State'),
            ('AL', 'Alabama'), ('AK', 'Alaska'), ('AZ', 'Arizona'), ('AR', 'Arkansas'),
            ('CA', 'California'), ('CO', 'Colorado'), ('CT', 'Connecticut'), ('DE', 'Delaware'),
            ('FL', 'Florida'), ('GA', 'Georgia'), ('HI', 'Hawaii'), ('ID', 'Idaho'),
            ('IL', 'Illinois'), ('IN', 'Indiana'), ('IA', 'Iowa'), ('KS', 'Kansas'),
            ('KY', 'Kentucky'), ('LA', 'Louisiana'), ('ME', 'Maine'), ('MD', 'Maryland'),
            ('MA', 'Massachusetts'), ('MI', 'Michigan'), ('MN', 'Minnesota'), ('MS', 'Mississippi'),
            ('MO', 'Missouri'), ('MT', 'Montana'), ('NE', 'Nebraska'), ('NV', 'Nevada'),
            ('NH', 'New Hampshire'), ('NJ', 'New Jersey'), ('NM', 'New Mexico'), ('NY', 'New York'),
            ('NC', 'North Carolina'), ('ND', 'North Dakota'), ('OH', 'Ohio'), ('OK', 'Oklahoma'),
            ('OR', 'Oregon'), ('PA', 'Pennsylvania'), ('RI', 'Rhode Island'), ('SC', 'South Carolina'),
            ('SD', 'South Dakota'), ('TN', 'Tennessee'), ('TX', 'Texas'), ('UT', 'Utah'),
            ('VT', 'Vermont'), ('VA', 'Virginia'), ('WA', 'Washington'), ('WV', 'West Virginia'),
            ('WI', 'Wisconsin'), ('WY', 'Wyoming')
        ]

        return context

    def get(self, request, *args, **kwargs):
        # Handle CSV export
        if request.GET.get('export') == 'csv':
            return self.export_csv()

        return super().get(request, *args, **kwargs)

    def export_csv(self):
        """Export search results to CSV"""
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="provider_search_results.csv"'

        writer = csv.writer(response)

        # Write header
        writer.writerow([
            'NPI', 'Type', 'Name/Organization', 'First Name', 'Last Name', 'Credential',
            'Address', 'City', 'State', 'ZIP Code', 'Phone', 'Primary Specialty',
            'Taxonomy Code', 'Enrollment Date', 'Last Update', 'Status'
        ])

        # Write data
        queryset = self.get_queryset()[:1000]  # Limit export to 1000 records

        for provider in queryset:
            # Get primary address
            address = provider.addresses.first()

            # Get primary specialty
            primary_taxonomy = provider.taxonomies.filter(is_primary=True).first()
            if not primary_taxonomy:
                primary_taxonomy = provider.taxonomies.first()

            writer.writerow([
                provider.npi,
                provider.entity_type,
                provider.full_name,
                provider.first_name or '',
                provider.last_name or '',
                provider.credential or '',
                f"{address.address_line_1} {address.address_line_2}".strip() if address else '',
                address.city if address else '',
                address.state if address else '',
                address.postal_code if address else '',
                provider.phone or '',
                primary_taxonomy.taxonomy_code.classification if primary_taxonomy else '',
                primary_taxonomy.taxonomy_code.code if primary_taxonomy else '',
                provider.enumeration_date or '',
                provider.last_update_date or '',
                'Inactive' if provider.deactivation_date else 'Active'
            ])

        return response


def provider_detail(request, npi):
    """Provider detail view (simplified)"""
    try:
        provider = Provider.objects.select_related().prefetch_related(
            'addresses', 'taxonomies__taxonomy_code'
        ).get(npi=npi)

        return render(request, 'search/provider_detail.html', {
            'provider': provider,
            'addresses': provider.addresses.all(),
            'taxonomies': provider.taxonomies.all(),
        })
    except Provider.DoesNotExist:
        return render(request, 'search/provider_not_found.html', {'npi': npi})