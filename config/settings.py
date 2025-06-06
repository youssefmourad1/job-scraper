"""
Configuration settings for Job Market Analyzer
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_CONFIG = {
    'main_db': str(DATA_DIR / "job_market.db"),
    'backup_db': str(DATA_DIR / "job_market_backup.db")
}

# Scraper configuration
SCRAPER_CONFIG = {
    'max_workers': 8,
    'max_pages_per_scraper': 50,
    'request_delay': 1,
    'timeout': 30,
    'retry_attempts': 3,
    'user_agents': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    ]
}

# Scraper URLs
SCRAPER_URLS = {
    'rekrute': {
        'base_url': 'https://www.rekrute.com/fr/offres.html',
        'search_params': {
            's': '3',  # Sort by date
            'p': '1',  # Page number
            'o': '1'   # Order
        }
    },
    'maroc_annonce': {
        'base_url': 'https://www.marocannonces.com/maroc/offres-emploi-b309.html',
        'search_params': {
            'pge': '1'  # Page number
        }
    }
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    'page_title': 'Professional Job Market Analyzer',
    'page_icon': '📊',
    'layout': 'wide',
    'theme': {
        'primaryColor': '#667eea',
        'backgroundColor': '#ffffff',
        'secondaryBackgroundColor': '#f0f2f6',
        'textColor': '#262730',
        'font': 'sans serif'
    }
}

# Analytics configuration
ANALYTICS_CONFIG = {
    'min_data_points': 10,
    'trend_analysis_days': 30,
    'emerging_jobs_threshold': 0.2,  # 20% growth rate
    'wordcloud_max_words': 100,
    'clustering_n_clusters': 5
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': str(LOGS_DIR / "job_market_analyzer.log"),
    'max_bytes': 10485760,  # 10MB
    'backup_count': 5
}

# Data cleaning configuration
DATA_CLEANING_CONFIG = {
    'text_fields': ['title', 'description', 'company', 'location'],
    'remove_duplicates_fields': ['title', 'company', 'url'],
    'date_formats': [
        '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y',
        '%d %m %Y', '%d.%m.%Y', '%Y/%m/%d',
        '%d-%b-%Y', '%d %b %Y', '%d/%b/%Y',
        '%B %d, %Y', '%d %B %Y', '%Y-%m-%d %H:%M:%S',
        '%d/%m/%Y %H:%M', '%d-%m-%Y %H:%M:%S'
    ],
    'stop_words_languages': ['french', 'english'],
    'min_text_length': 3
} 