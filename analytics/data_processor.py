"""
Data processing and cleaning for Job Market Analyzer
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from config.database import DatabaseManager
from config.settings import DATA_CLEANING_CONFIG, ANALYTICS_CONFIG
from utils.helpers import (
    clean_text, extract_salary_range, parse_date, 
    extract_skills_from_text, categorize_experience
)

logger = logging.getLogger('analytics.data_processor')

class DataProcessor:
    """
    Advanced data processing and cleaning for job market data
    """
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf_vectorizer = None
        self._setup_nltk()
        
    def _setup_nltk(self):
        """Setup NLTK resources"""
        try:
            self.stop_words = set(stopwords.words('french')) | set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            nltk.download('stopwords')
            nltk.download('wordnet')
            nltk.download('punkt')
            self.stop_words = set(stopwords.words('french')) | set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
    
    def load_and_clean_data(self, filters: Dict = None) -> pd.DataFrame:
        """Load data from database and apply comprehensive cleaning"""
        logger.info("Loading and cleaning job market data")
        
        # Load raw data
        df = self.db_manager.get_jobs_dataframe(filters)
        
        if df.empty:
            logger.warning("No data found in database")
            return df
        
        logger.info(f"Loaded {len(df)} job records")
        
        # Apply cleaning steps
        df = self._clean_text_fields(df)
        df = self._standardize_dates(df)
        df = self._process_salary_data(df)
        df = self._categorize_fields(df)
        df = self._extract_features(df)
        df = self._remove_duplicates(df)
        df = self._handle_missing_values(df)
        
        logger.info(f"Cleaned data: {len(df)} records remaining")
        return df
    
    def _clean_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize text fields"""
        logger.debug("Cleaning text fields")
        
        for field in DATA_CLEANING_CONFIG['text_fields']:
            if field in df.columns:
                df[field] = df[field].astype(str).apply(clean_text)
                # Remove very short texts
                min_length = DATA_CLEANING_CONFIG['min_text_length']
                df.loc[df[field].str.len() < min_length, field] = ''
        
        return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize date formats"""
        logger.debug("Standardizing date formats")
        
        date_columns = ['date_posted', 'date_scraped', 'created_at', 'updated_at']
        
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Add derived date features
        if 'date_posted' in df.columns:
            df['days_since_posted'] = (datetime.now() - df['date_posted']).dt.days
            df['posting_month'] = df['date_posted'].dt.month
            df['posting_weekday'] = df['date_posted'].dt.dayofweek
            df['posting_quarter'] = df['date_posted'].dt.quarter
        
        return df
    
    def _process_salary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and normalize salary information"""
        logger.debug("Processing salary data")
        
        if 'salary' in df.columns:
            # Extract numeric salary ranges
            salary_ranges = df['salary'].apply(extract_salary_range)
            df['salary_min_extracted'] = salary_ranges.apply(lambda x: x['min'])
            df['salary_max_extracted'] = salary_ranges.apply(lambda x: x['max'])
            
            # Calculate average salary where possible
            df['salary_avg'] = df[['salary_min_extracted', 'salary_max_extracted']].mean(axis=1)
            
            # Categorize salary ranges
            df['salary_category'] = pd.cut(
                df['salary_avg'],
                bins=[0, 5000, 10000, 20000, 50000, np.inf],
                labels=['Low', 'Medium', 'High', 'Very High', 'Executive'],
                include_lowest=True
            )
        
        return df
    
    def _categorize_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize and standardize categorical fields"""
        logger.debug("Categorizing fields")
        
        # Standardize experience levels
        if 'experience' in df.columns:
            df['experience_category'] = df['experience'].apply(categorize_experience)
        
        # Standardize education levels
        if 'education_level' in df.columns:
            df['education_category'] = df['education_level'].apply(self._categorize_education)
        
        # Standardize contract types
        if 'contract_type' in df.columns:
            df['contract_category'] = df['contract_type'].apply(self._categorize_contract)
        
        # Standardize company sizes (if available)
        if 'company_type' in df.columns:
            df['company_size_category'] = df['company_type'].apply(self._categorize_company_size)
        
        return df
    
    def _categorize_education(self, education_str: str) -> str:
        """Categorize education levels"""
        if not education_str:
            return "Unknown"
        
        education_lower = education_str.lower()
        
        if any(word in education_lower for word in ['doctorat', 'phd', 'doctorate']):
            return "Doctorate"
        elif any(word in education_lower for word in ['master', 'mastère', 'ingénieur']):
            return "Master"
        elif any(word in education_lower for word in ['licence', 'bachelor', 'bac+3']):
            return "Bachelor"
        elif any(word in education_lower for word in ['bac+2', 'dut', 'bts']):
            return "Associate"
        elif any(word in education_lower for word in ['bac', 'secondaire']):
            return "High School"
        else:
            return "Other"
    
    def _categorize_contract(self, contract_str: str) -> str:
        """Categorize contract types"""
        if not contract_str:
            return "Unknown"
        
        contract_lower = contract_str.lower()
        
        if any(word in contract_lower for word in ['cdi', 'permanent', 'indefinite']):
            return "Permanent"
        elif any(word in contract_lower for word in ['cdd', 'temporary', 'fixed']):
            return "Temporary"
        elif any(word in contract_lower for word in ['stage', 'internship']):
            return "Internship"
        elif any(word in contract_lower for word in ['freelance', 'consultant', 'independent']):
            return "Freelance"
        elif any(word in contract_lower for word in ['part-time', 'temps partiel']):
            return "Part-time"
        else:
            return "Other"
    
    def _categorize_company_size(self, company_type_str: str) -> str:
        """Categorize company sizes"""
        if not company_type_str:
            return "Unknown"
        
        company_lower = company_type_str.lower()
        
        if any(word in company_lower for word in ['startup', 'start-up']):
            return "Startup"
        elif any(word in company_lower for word in ['multinational', 'international', 'global']):
            return "Large Enterprise"
        elif any(word in company_lower for word in ['pme', 'sme', 'medium']):
            return "Medium Enterprise"
        elif any(word in company_lower for word in ['small', 'petite']):
            return "Small Enterprise"
        else:
            return "Unknown"
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced features from text and other fields"""
        logger.debug("Extracting advanced features")
        
        # Extract skills from descriptions
        if 'description' in df.columns:
            df['extracted_skills'] = df['description'].apply(extract_skills_from_text)
            df['skills_count'] = df['extracted_skills'].apply(len)
            
            # Create binary features for common skills
            common_skills = self._get_common_skills(df)
            for skill in common_skills:
                df[f'has_{skill.replace(" ", "_")}'] = df['extracted_skills'].apply(
                    lambda skills: skill.lower() in [s.lower() for s in skills]
                )
        
        # Extract text features
        if 'title' in df.columns:
            df['title_length'] = df['title'].str.len()
            df['title_word_count'] = df['title'].str.split().str.len()
        
        if 'description' in df.columns:
            df['description_length'] = df['description'].str.len()
            df['description_word_count'] = df['description'].str.split().str.len()
        
        # Location-based features
        if 'location' in df.columns:
            df['is_casablanca'] = df['location'].str.contains('Casablanca', case=False, na=False)
            df['is_rabat'] = df['location'].str.contains('Rabat', case=False, na=False)
            df['is_marrakech'] = df['location'].str.contains('Marrakech', case=False, na=False)
        
        return df
    
    def _get_common_skills(self, df: pd.DataFrame, top_n: int = 20) -> List[str]:
        """Get most common skills from the dataset"""
        all_skills = []
        for skills_list in df['extracted_skills'].dropna():
            all_skills.extend(skills_list)
        
        skill_counts = pd.Series(all_skills).value_counts()
        return skill_counts.head(top_n).index.tolist()
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate records based on configured fields"""
        logger.debug("Removing duplicates")
        
        duplicate_fields = DATA_CLEANING_CONFIG['remove_duplicates_fields']
        available_fields = [f for f in duplicate_fields if f in df.columns]
        
        if available_fields:
            initial_count = len(df)
            df = df.drop_duplicates(subset=available_fields, keep='last')
            removed_count = initial_count - len(df)
            logger.info(f"Removed {removed_count} duplicate records")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        logger.debug("Handling missing values")
        
        # Convert categorical columns back to object to avoid category issues
        categorical_columns = df.select_dtypes(include=['category']).columns
        for col in categorical_columns:
            df[col] = df[col].astype('object')
        
        # Fill missing categorical/object values
        text_columns = df.select_dtypes(include=['object']).columns
        for col in text_columns:
            if col in ['location', 'sector', 'fonction', 'company', 'title']:
                df[col] = df[col].fillna('Unknown')
            elif col in ['source', 'contract_type', 'education_level']:
                df[col] = df[col].fillna('Not Specified')
            else:
                df[col] = df[col].fillna('Unknown')
        
        # Fill missing numerical values
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if 'salary' in col:
                df[col] = df[col].fillna(0)
            elif 'count' in col or 'length' in col:
                df[col] = df[col].fillna(0)
            else:
                # Use median for other numerical columns, but handle edge cases
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
        
        return df
    
    def create_text_features(self, df: pd.DataFrame, text_column: str = 'description') -> pd.DataFrame:
        """Create TF-IDF features from text data"""
        logger.debug(f"Creating text features from {text_column}")
        
        if text_column not in df.columns:
            return df
        
        # Prepare text data
        texts = df[text_column].fillna('').astype(str)
        
        # Create TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=list(self.stop_words),
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Create feature names
        feature_names = [f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out()]
        
        # Convert to DataFrame and concatenate
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)
        df = pd.concat([df, tfidf_df], axis=1)
        
        logger.info(f"Created {len(feature_names)} TF-IDF features")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for machine learning"""
        logger.debug("Encoding categorical features")
        
        categorical_columns = [
            'source', 'sector', 'fonction', 'experience_category',
            'education_category', 'contract_category', 'company_size_category',
            'salary_category'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                # Convert to string and handle missing values
                df[col] = df[col].astype(str).fillna('Unknown')
                
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    # Handle new categories safely
                    unique_values = df[col].unique()
                    known_values = set(self.label_encoders[col].classes_)
                    
                    # Add new categories to the encoder
                    new_categories = set(unique_values) - known_values
                    if new_categories:
                        all_categories = list(known_values) + list(new_categories)
                        self.label_encoders[col].classes_ = np.array(all_categories)
                    
                    # Transform with proper error handling
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                    except ValueError:
                        # Fallback: refit the encoder with all current data
                        df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def get_processed_data_summary(self, df: pd.DataFrame) -> Dict:
        """Get summary statistics of processed data"""
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['date_posted'].min() if 'date_posted' in df.columns else None,
                'end': df['date_posted'].max() if 'date_posted' in df.columns else None
            },
            'unique_companies': df['company'].nunique() if 'company' in df.columns else 0,
            'unique_sectors': df['sector'].nunique() if 'sector' in df.columns else 0,
            'unique_locations': df['location'].nunique() if 'location' in df.columns else 0,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        return summary 