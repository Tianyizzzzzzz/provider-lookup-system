# Healthcare Provider Lookup System

A comprehensive web-based application for searching and retrieving healthcare provider information from the CMS NPPES (National Plan and Provider Enumeration System) database.

##  Features

- **Comprehensive Search**: Search by NPI number, provider name, or organization
- **Location-based Filtering**: Filter by city and state
- **Detailed Provider Information**: Display 16 columns of provider data including contact info, specialties, and credentials
- **Large Dataset Support**: Handles 8.9+ million provider records
- **Export Functionality**: Export search results to CSV
- **Responsive Design**: Works on desktop and mobile devices
- **High Performance**: PostgreSQL backend with optimized queries

##  Technology Stack

- **Backend**: Django 5.2.4, Python 3.13
- **Database**: PostgreSQL with full-text search capabilities
- **Frontend**: Bootstrap 5.3.0, HTML5, JavaScript
- **Data Processing**: Pandas, custom ETL pipeline
- **Deployment**: Docker support included

##  Database Schema

The system includes optimized models for:
- **Provider**: Core provider information (NPI, names, contact details)
- **ProviderAddress**: Multiple address types (practice locations, mailing addresses)
- **ProviderTaxonomy**: Medical specialties and classifications
- **TaxonomyCode**: Standardized medical specialty codes

##  Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/provider-lookup-system.git
   cd provider-lookup-system
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd provider_lookup_web
   pip install -r requirements.txt
   ```

4. **Database setup**
   ```bash
   # Create PostgreSQL database
   createdb provider_lookup
   
   # Configure environment variables
   cp ../.env.example .env
   # Edit .env with your database credentials
   ```

5. **Run migrations**
   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```

6. **Start development server**
   ```bash
   python manage.py runserver
   ```

Visit `http://127.0.0.1:8000` to access the application.

## 🐳 Docker Deployment

1. **Using Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - Web interface: `http://localhost:8000`
   - Admin interface: `http://localhost:8000/admin`

## 📂 Project Structure

```
provider-lookup-system/
├── provider_lookup_web/          # Django web application
│   ├── apps/                     # Django applications
│   │   ├── providers/            # Provider models and admin
│   │   ├── search/               # Search functionality
│   │   └── common/               # Shared utilities
│   ├── config/                   # Django settings and URLs
│   ├── templates/                # HTML templates
│   ├── static/                   # CSS, JS, images
│   └── requirements.txt          # Python dependencies
├── scripts/                      # Data processing scripts
│   ├── data_pipeline/            # ETL pipeline for NPPES data
│   └── utils/                    # Utility functions
├── tests/                        # Test suites
├── docs/                         # Documentation
├── docker-compose.yml            # Docker configuration
└── README.md                     # This file
```

## 🔍 Usage

### Basic Search
1. Select search type (All Fields, NPI Number, or Provider Name)
2. Enter search query
3. Optionally filter by city and state
4. Click "Search" to view results

### Advanced Features
- **Export Results**: Click "Export CSV" to download search results
- **Pagination**: Navigate through large result sets
- **Responsive Table**: Hover over cells to see full content

### Search Types
- **NPI Search**: Enter 10-digit National Provider Identifier for exact match
- **Name Search**: Partial matching for provider names and organizations
- **Location Search**: Filter by city and/or state

##  Data Sources

This system uses data from:
- **CMS NPPES Database**: National Provider Identifier registry
- **NUCC Taxonomy**: Healthcare provider taxonomy classifications

##  Configuration

### Environment Variables
Key settings in `.env` file:
```
POSTGRES_DB=provider_lookup
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
SECRET_KEY=your-secret-key
DEBUG=True
```

### Database Optimization
The system includes several performance optimizations:
- Database indexes on frequently searched fields
- Query optimization with select_related and prefetch_related
- Full-text search capabilities using PostgreSQL

## 🧪 Testing

Run the test suite:
```bash
python manage.py test
```

## 📈 Performance

- **Database Size**: 8.9+ million provider records
- **Search Performance**: Sub-second response times for most queries
- **Concurrent Users**: Optimized for multiple simultaneous searches
- **Export Capability**: Handle up to 1000 records per CSV export

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CMS for providing the NPPES database
- NUCC for healthcare provider taxonomy classifications
- Django and PostgreSQL communities

## 📞 Support

For support, please open an issue on GitHub or contact [tianyizex1@gmail.com].

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with applicable healthcare data regulations when deploying in production environments.