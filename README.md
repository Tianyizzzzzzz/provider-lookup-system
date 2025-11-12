# Healthcare Provider Lookup System

An AI-enhanced healthcare provider search system using data mining techniques to assess and classify provider data quality in the National Plan and Provider Enumeration System (NPPES) database.

## 🎯 Project Overview

This project applies clustering and classification algorithms to analyze 120,706 healthcare provider records from the NPPES database. The system identifies data quality patterns and enables intelligent prioritization of high-quality provider records in search results.

## 📊 Key Features

- **Data Quality Assessment**: Automated classification of provider records into quality tiers
- **Clustering Analysis**: K-means clustering identifying 3 distinct provider groups
- **Multi-Model Classification**: Comparison of 4 ML algorithms (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- **Comprehensive Evaluation**: Complete metrics including accuracy, precision, recall, F1, ROC/AUC
- **Django Integration**: Web-based provider lookup system with PostgreSQL backend

## 🔬 Technical Approach

### Phase 1: Exploratory Data Analysis
- Dataset: 8.96 million NPPES records
- Sample: 120,706 high-quality providers
- Feature engineering: 15 initial features → 9 with variance

### Phase 2: Clustering Analysis
- Algorithm: MiniBatchKMeans (k=3)
- Silhouette Score: 0.313
- Runtime: ~3 minutes
- Results: 3 quality-based provider segments

### Phase 3: Classification Modeling
- Models: Logistic Regression, Random Forest, Gradient Boosting, SVM
- Accuracy: 99.98-100%
- Key Finding: Contact information completeness is the primary quality discriminator

## 📈 Results Summary

| Cluster | Size | Contact Completeness | Overall Quality Score |
|---------|------|---------------------|----------------------|
| Cluster 0 | 33.2% | 100% | 75.0 |
| Cluster 1 | 24.0% | 61.1% | 60.5 |
| Cluster 2 | 42.8% | 100% | 83.7 |

All classification models achieved >99.9% accuracy in predicting cluster membership.

## 🛠️ Technologies Used

- **Backend**: Django 5.2, PostgreSQL
- **Data Processing**: pandas, NumPy
- **Machine Learning**: scikit-learn (MiniBatchKMeans, Logistic Regression, Random Forest, Gradient Boosting, SVM)
- **Visualization**: matplotlib, seaborn
- **Data Source**: NPPES NPI Registry (8.96M records, 10.7 GB)

## 📁 Project Structure
```
provider_lookup/
├── provider_lookup_web/      # Django application
│   ├── apps/providers/       # Provider models and views
│   └── config/               # Django settings
├── data_mining/              # Analysis scripts
│   ├── 02_clustering_final.py
│   ├── 03_classification_final.py
│   ├── clustering_results/   # Phase 2 outputs
│   └── classification_results/ # Phase 3 outputs
└── docs/                     # Documentation
```

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.11+
PostgreSQL 15+
10.7 GB disk space for dataset
```

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/provider_lookup.git
cd provider_lookup

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r provider_lookup_web/requirements.txt

# Setup database
python provider_lookup_web/manage.py migrate
```

### Run Analysis
```bash
# Phase 2: Clustering Analysis
python data_mining/02_clustering_final.py

# Phase 3: Classification Modeling
python data_mining/03_classification_final.py
```

## 📊 Visualizations

The project generates 7 publication-quality visualizations:

**Phase 2 - Clustering:**
- Optimal cluster determination (Elbow + Silhouette)
- PCA projection of clusters
- Cluster size distribution

**Phase 3 - Classification:**
- Model performance comparison
- Confusion matrix (best model)
- Feature importance analysis
- ROC curves (one-vs-rest)

## 🎓 Academic Context

This project was developed for MATH 4720 (Statistical Data Mining) at [Your University]. It demonstrates practical applications of:
- Unsupervised learning (clustering)
- Supervised learning (classification)
- Feature engineering and selection
- Model evaluation and comparison
- Large-scale data processing

## 📄 Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md) - Detailed project description
- [Methodology](docs/METHODOLOGY.md) - Technical approach and algorithms
- [Results Summary](docs/RESULTS_SUMMARY.md) - Comprehensive results analysis

## 🤝 Contributing

This is an academic project. For questions or suggestions, please open an issue.

## 📧 Contact

Tianyi Zhang - tianyizex1@gmail.com

Project Link: https://github.com/Tianyizzzzzzz/provider-lookup-system

## 📜 License

This project is for educational purposes.

## 🙏 Acknowledgments

- NPPES for providing the healthcare provider dataset
- Professor Yousef for guidance and feedback
- MATH 4720 course materials and resources