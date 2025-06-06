"""
Helper functions for Job Market Analyzer
"""
import re
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import unicodedata
import random
from config.settings import DATA_CLEANING_CONFIG, SCRAPER_CONFIG

def clean_text(text: str) -> str:
    """Clean and normalize text data"""
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and handle unicode
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.,;:!?\-()]', '', text)
    
    return text

def extract_salary_range(salary_str: str) -> Dict[str, Optional[int]]:
    """Extract minimum and maximum salary from salary string"""
    if not salary_str or pd.isna(salary_str):
        return {'min': None, 'max': None}
    
    # Remove common currency symbols and words
    clean_salary = re.sub(r'[^\d\-\s]', '', str(salary_str))
    
    # Look for ranges (e.g., "5000-8000", "5000 - 8000")
    range_match = re.search(r'(\d+)\s*-\s*(\d+)', clean_salary)
    if range_match:
        return {
            'min': int(range_match.group(1)),
            'max': int(range_match.group(2))
        }
    
    # Look for single number
    single_match = re.search(r'(\d+)', clean_salary)
    if single_match:
        salary_value = int(single_match.group(1))
        return {'min': salary_value, 'max': salary_value}
    
    return {'min': None, 'max': None}

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string with multiple format support"""
    if not date_str or pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    # Try different date formats
    for date_format in DATA_CLEANING_CONFIG['date_formats']:
        try:
            return datetime.strptime(date_str, date_format)
        except ValueError:
            continue
    
    # Try pandas date parser as fallback
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return None

def extract_skills_from_text(text: str, skill_keywords: List[str] = None) -> List[str]:
    """Extract skills from job description or requirements"""
    if not text:
        return []
    
    # Default skill keywords (can be expanded)
    default_skills = [
        'python', 'java', 'javascript', 'php', 'sql', 'mysql', 'postgresql',
        'mongodb', 'redis', 'elasticsearch', 'docker', 'kubernetes', 'aws',
        'azure', 'gcp', 'react', 'angular', 'vue', 'nodejs', 'spring',
        'django', 'flask', 'laravel', 'symfony', 'tensorflow', 'pytorch',
        'machine learning', 'deep learning', 'data science', 'analytics',
        'tableau', 'power bi', 'excel', 'scrum', 'agile', 'git', 'github',
        'jenkins', 'ci/cd', 'linux', 'windows', 'macos'
    ]
    
    skills_to_check = skill_keywords or default_skills
    found_skills = []
    
    text_lower = text.lower()
    
    for skill in skills_to_check:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    
    return found_skills

def get_random_user_agent() -> str:
    """Get a random user agent string"""
    return random.choice(SCRAPER_CONFIG['user_agents'])

def calculate_growth_rate(current: float, previous: float) -> float:
    """Calculate growth rate percentage"""
    if previous == 0:
        return 0.0
    return ((current - previous) / previous) * 100

def categorize_experience(experience_str: str) -> str:
    """Categorize experience level"""
    if not experience_str:
        return "Unknown"
    
    experience_lower = experience_str.lower()
    
    if any(word in experience_lower for word in ['junior', 'débutant', 'entry', '0-2']):
        return "Junior"
    elif any(word in experience_lower for word in ['senior', 'expérimenté', '5+', '7+']):
        return "Senior"
    elif any(word in experience_lower for word in ['lead', 'manager', 'chef', 'director']):
        return "Leadership"
    elif any(word in experience_lower for word in ['mid', 'moyen', '2-5', '3-5']):
        return "Mid-level"
    else:
        return "Unknown"

def validate_url(url: str) -> bool:
    """Validate if URL is properly formatted"""
    if not url:
        return False
    
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(url) is not None

def format_number(number: int) -> str:
    """Format number with thousand separators"""
    if number is None:
        return "0"
    return f"{number:,}"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length-3] + "..."

def get_date_range_filter(days: int) -> Dict[str, datetime]:
    """Get date range filter for the last N days"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return {
        'date_from': start_date.strftime('%Y-%m-%d'),
        'date_to': end_date.strftime('%Y-%m-%d')
    }

def detect_language(text: str) -> str:
    """Simple language detection (French/English)"""
    if not text:
        return "unknown"
    
    french_words = ['le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'pour', 'avec', 'dans']
    english_words = ['the', 'and', 'or', 'for', 'with', 'in', 'of', 'to', 'a', 'an']
    
    text_lower = text.lower()
    
    french_count = sum(1 for word in french_words if word in text_lower)
    english_count = sum(1 for word in english_words if word in text_lower)
    
    if french_count > english_count:
        return "french"
    elif english_count > french_count:
        return "english"
    else:
        return "unknown" 