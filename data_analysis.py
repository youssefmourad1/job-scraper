import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from datetime import datetime
import sqlite3
import streamlit as st
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class DataAnalysis:
    def __init__(self, rekrute_db: str = "rekrute_jobs.db", maroc_annonce_db: str = "maroc_annonce_jobs.db"):
        """
        Initialize the DataAnalysis class with database connections and NLTK setup
        """
        self.rekrute_db = rekrute_db
        self.maroc_annonce_db = maroc_annonce_db
        self.df_combined = None
        self._setup_nltk()
        self._load_data()
        
    def _setup_nltk(self):
        """Setup NLTK resources"""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('french'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _load_data(self):
        """Load and combine data from both databases"""
        dfs = []
        
        # Load Rekrute data
        try:
            conn = sqlite3.connect(self.rekrute_db)
            df_rekrute = pd.read_sql("SELECT * FROM jobs", conn)
            df_rekrute['source'] = 'Rekrute'
            dfs.append(df_rekrute)
            conn.close()
        except Exception as e:
            st.warning(f"Could not load Rekrute data: {str(e)}")
        
        # Load MarocAnnonce data
        try:
            conn = sqlite3.connect(self.maroc_annonce_db)
            df_maroc = pd.read_sql("SELECT * FROM jobs", conn)
            df_maroc['source'] = 'MarocAnnonce'
            dfs.append(df_maroc)
            conn.close()
        except Exception as e:
            st.warning(f"Could not load MarocAnnonce data: {str(e)}")
        
        if dfs:
            self.df_combined = pd.concat(dfs, ignore_index=True)
            self._preprocess_data()
        else:
            st.error("No data could be loaded from either database")
    
    def _preprocess_data(self):
        """Preprocess the combined dataset"""
        if self.df_combined is None:
            return
            
        # Convert date columns to datetime
        date_columns = ['pub_start', 'pub_end', 'date_posted']
        for col in date_columns:
            if col in self.df_combined.columns:
                self.df_combined[col] = pd.to_datetime(self.df_combined[col], errors='coerce')
        
        # Clean text columns
        text_columns = ['title', 'description', 'company_desc', 'mission_desc']
        for col in text_columns:
            if col in self.df_combined.columns:
                self.df_combined[col] = self.df_combined[col].fillna('').astype(str)
                self.df_combined[col] = self.df_combined[col].apply(self._clean_text)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def get_basic_stats(self) -> Dict:
        """Get basic statistics about the dataset"""
        if self.df_combined is None:
            return {}
            
        stats = {
            'total_jobs': len(self.df_combined),
            'unique_companies': self.df_combined['company'].nunique() if 'company' in self.df_combined.columns else 0,
            'unique_locations': self.df_combined['location'].nunique() if 'location' in self.df_combined.columns else 0,
            'unique_sectors': self.df_combined['sector'].nunique() if 'sector' in self.df_combined.columns else 0,
            'date_range': {
                'start': self.df_combined['pub_start'].min() if 'pub_start' in self.df_combined.columns else None,
                'end': self.df_combined['pub_start'].max() if 'pub_start' in self.df_combined.columns else None
            }
        }
        return stats
    
    def analyze_job_distribution(self) -> Dict[str, go.Figure]:
        """Analyze job distribution across different dimensions"""
        if self.df_combined is None:
            return {}
            
        figures = {}
        
        # Source distribution
        if 'source' in self.df_combined.columns:
            source_counts = self.df_combined['source'].value_counts()
            fig_source = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Job Distribution by Source",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            figures['source_distribution'] = fig_source
        
        # Location analysis
        if 'location' in self.df_combined.columns:
            location_counts = self.df_combined['location'].value_counts().head(10)
            fig_location = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="Top 10 Job Locations",
                color=location_counts.values,
                color_continuous_scale="plasma"
            )
            figures['location_distribution'] = fig_location
        
        # Sector analysis
        if 'sector' in self.df_combined.columns:
            sector_counts = self.df_combined['sector'].value_counts().head(10)
            fig_sector = px.bar(
                x=sector_counts.values,
                y=sector_counts.index,
                orientation='h',
                title="Top 10 Job Sectors",
                color=sector_counts.values,
                color_continuous_scale="viridis"
            )
            figures['sector_distribution'] = fig_sector
        
        return figures
    
    def analyze_skills_demand(self) -> Dict[str, go.Figure]:
        """Analyze skills and requirements demand"""
        if self.df_combined is None:
            return {}
            
        figures = {}
        
        # Function analysis
        if 'fonction' in self.df_combined.columns:
            func_counts = self.df_combined['fonction'].value_counts().head(15)
            fig_func = px.treemap(
                names=func_counts.index,
                values=func_counts.values,
                title="Job Functions Distribution (Top 15)"
            )
            figures['function_distribution'] = fig_func
        
        # Experience level analysis
        if 'experience' in self.df_combined.columns:
            exp_counts = self.df_combined['experience'].value_counts()
            fig_exp = px.pie(
                values=exp_counts.values,
                names=exp_counts.index,
                title="Experience Level Distribution"
            )
            figures['experience_distribution'] = fig_exp
        
        # Education level analysis
        if 'study_level' in self.df_combined.columns:
            edu_counts = self.df_combined['study_level'].value_counts()
            fig_edu = px.pie(
                values=edu_counts.values,
                names=edu_counts.index,
                title="Education Level Distribution"
            )
            figures['education_distribution'] = fig_edu
        
        return figures
    
    def analyze_market_trends(self) -> Dict[str, go.Figure]:
        """Analyze market trends and temporal patterns"""
        if self.df_combined is None:
            return {}
            
        figures = {}
        
        # Time series analysis
        if 'pub_start' in self.df_combined.columns:
            # Daily job postings
            daily_posts = self.df_combined.groupby(self.df_combined['pub_start'].dt.date).size()
            fig_daily = px.line(
                x=daily_posts.index,
                y=daily_posts.values,
                title="Daily Job Postings Trend",
                labels={'x': 'Date', 'y': 'Number of Posts'}
            )
            figures['daily_trend'] = fig_daily
            
            # Monthly job postings by sector
            if 'sector' in self.df_combined.columns:
                monthly_sector = self.df_combined.groupby([
                    self.df_combined['pub_start'].dt.to_period('M'),
                    'sector'
                ]).size().reset_index(name='count')
                monthly_sector['pub_start'] = monthly_sector['pub_start'].astype(str)
                
                fig_monthly = px.line(
                    monthly_sector,
                    x='pub_start',
                    y='count',
                    color='sector',
                    title="Monthly Job Postings by Sector",
                    labels={'pub_start': 'Month', 'count': 'Number of Posts'}
                )
                figures['monthly_sector_trend'] = fig_monthly
        
        return figures
    
    def generate_skill_wordcloud(self) -> go.Figure:
        """Generate word cloud from job descriptions"""
        if self.df_combined is None or 'description' not in self.df_combined.columns:
            return None
            
        # Combine all descriptions
        text = ' '.join(self.df_combined['description'].fillna(''))
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        # Convert to plotly figure
        fig = go.Figure()
        fig.add_trace(go.Image(z=wordcloud.to_array()))
        fig.update_layout(
            title="Most Common Skills and Requirements",
            xaxis_showticklabels=False,
            yaxis_showticklabels=False
        )
        
        return fig
    
    def analyze_salary_trends(self) -> Dict[str, go.Figure]:
        """Analyze salary trends and distributions"""
        if self.df_combined is None:
            return {}
            
        figures = {}
        
        if 'salary' in self.df_combined.columns:
            # Extract numeric salary values
            def extract_salary(salary_str):
                if pd.isna(salary_str):
                    return None
                # Extract numbers from salary string
                numbers = re.findall(r'\d+', str(salary_str))
                if numbers:
                    return int(numbers[0])
                return None
            
            self.df_combined['salary_numeric'] = self.df_combined['salary'].apply(extract_salary)
            
            # Salary distribution by sector
            if 'sector' in self.df_combined.columns:
                fig_salary_sector = px.box(
                    self.df_combined,
                    x='sector',
                    y='salary_numeric',
                    title="Salary Distribution by Sector",
                    labels={'salary_numeric': 'Salary', 'sector': 'Sector'}
                )
                figures['salary_by_sector'] = fig_salary_sector
            
            # Salary distribution by experience
            if 'experience' in self.df_combined.columns:
                fig_salary_exp = px.box(
                    self.df_combined,
                    x='experience',
                    y='salary_numeric',
                    title="Salary Distribution by Experience Level",
                    labels={'salary_numeric': 'Salary', 'experience': 'Experience Level'}
                )
                figures['salary_by_experience'] = fig_salary_exp
        
        return figures
    
    def analyze_emerging_jobs(self) -> pd.DataFrame:
        """Analyze emerging job trends"""
        if self.df_combined is None:
            return pd.DataFrame()
            
        # Calculate job growth rate
        if 'pub_start' in self.df_combined.columns and 'fonction' in self.df_combined.columns:
            # Group by function and month
            monthly_jobs = self.df_combined.groupby([
                self.df_combined['pub_start'].dt.to_period('M'),
                'fonction'
            ]).size().reset_index(name='count')
            
            # Calculate growth rate
            monthly_jobs['pub_start'] = monthly_jobs['pub_start'].astype(str)
            monthly_jobs = monthly_jobs.sort_values(['fonction', 'pub_start'])
            monthly_jobs['growth_rate'] = monthly_jobs.groupby('fonction')['count'].pct_change()
            
            # Get the most recent month's growth rates
            latest_month = monthly_jobs['pub_start'].max()
            emerging_jobs = monthly_jobs[monthly_jobs['pub_start'] == latest_month].sort_values(
                'growth_rate',
                ascending=False
            ).head(10)
            
            return emerging_jobs
        
        return pd.DataFrame()
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate a comprehensive analysis report"""
        if self.df_combined is None:
            return {}
            
        report = {
            'basic_stats': self.get_basic_stats(),
            'job_distribution': self.analyze_job_distribution(),
            'skills_demand': self.analyze_skills_demand(),
            'market_trends': self.analyze_market_trends(),
            'salary_trends': self.analyze_salary_trends(),
            'emerging_jobs': self.analyze_emerging_jobs(),
            'skill_wordcloud': self.generate_skill_wordcloud()
        }
        
        return report

class JobMarketAnalyzer:
    """
    Advanced Job Market Analysis Class
    Provides comprehensive EDA, skills analysis, trend prediction, and market insights
    """
    
    def __init__(self, db_path: str = "data/job_market.db"):
        """
        Initialize the Job Market Analyzer
        
        Args:
            db_path: Path to the SQLite database containing job data
        """
        self.db_path = db_path
        self.df = None
        self.processed_df = None
        self.skills_data = None
        self.trends_data = None
        self.clusters = None
        self.prediction_models = {}
        
        # Initialize NLTK
        self._setup_nltk()
        
        # Load and preprocess data
        self._load_data()
        if self.df is not None:
            self._preprocess_data()
    
    def _setup_nltk(self):
        """Setup NLTK resources for text processing"""
        try:
            import nltk
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        
        self.stop_words_fr = set(stopwords.words('french'))
        self.stop_words_en = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Common tech skills and keywords
        self.tech_skills = {
            'programming': ['python', 'java', 'javascript', 'php', 'c++', 'c#', 'sql', 'html', 'css', 'react', 'angular', 'vue'],
            'data_science': ['machine learning', 'deep learning', 'ai', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn'],
            'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'cloud computing'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
            'tools': ['git', 'jenkins', 'jira', 'confluence', 'slack']
        }
    
    def _load_data(self):
        """Load job data from database"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT title, company, location, sector, fonction,
                   experience, education_level, contract_type,
                   salary, salary_min, salary_max,
                   description, requirements, source,
                   is_active, view_count, likes, date_scraped,
                   pub_start, pub_end
            FROM jobs 
            WHERE is_active = 1
            """
            
            self.df = pd.read_sql_query(query, conn)
            conn.close()
            
            print(f"✅ Loaded {len(self.df)} job records from database")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self.df = None
    
    def _preprocess_data(self):
        """Comprehensive data preprocessing and cleaning"""
        if self.df is None:
            return
        
        print("🔄 Starting data preprocessing...")
        
        # Create a copy for processing
        self.processed_df = self.df.copy()
        
        # 1. Date processing
        date_columns = ['date_scraped', 'pub_start', 'pub_end']
        for col in date_columns:
            if col in self.processed_df.columns:
                self.processed_df[col] = pd.to_datetime(self.processed_df[col], errors='coerce')
        
        # 2. Text cleaning and normalization
        text_columns = ['title', 'description', 'requirements', 'company']
        for col in text_columns:
            if col in self.processed_df.columns:
                self.processed_df[col] = self.processed_df[col].fillna('').astype(str)
                self.processed_df[f'{col}_clean'] = self.processed_df[col].apply(self._clean_text)
        
        # 3. Salary processing
        self._process_salary_data()
        
        # 4. Experience level standardization
        self._standardize_experience()
        
        # 5. Location standardization
        self._standardize_locations()
        
        # 6. Sector classification
        self._classify_sectors()
        
        # 7. Extract skills from descriptions
        self._extract_skills()
        
        print("✅ Data preprocessing completed")
    
    def _clean_text(self, text: str) -> str:
        """Advanced text cleaning"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        import re
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove special characters but keep French accents
        text = re.sub(r'[^\w\sàâäéèêëïîôöùûüÿç]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _process_salary_data(self):
        """Process and extract salary information"""
        if self.processed_df is None:
            return
            
        def extract_salary_range(salary_str):
            if pd.isna(salary_str):
                return None, None
            
            import re
            # Extract numbers from salary string
            numbers = re.findall(r'\d+', str(salary_str))
            if len(numbers) >= 2:
                return int(numbers[0]), int(numbers[1])
            elif len(numbers) == 1:
                return int(numbers[0]), int(numbers[0])
            return None, None
        
        # Process salary strings
        if 'salary' in self.processed_df.columns:
            salary_ranges = self.processed_df['salary'].apply(extract_salary_range)
            self.processed_df['salary_min_extracted'] = [r[0] if r else None for r in salary_ranges]
            self.processed_df['salary_max_extracted'] = [r[1] if r else None for r in salary_ranges]
        
        # Combine with existing salary columns
        if 'salary_min' not in self.processed_df.columns:
            self.processed_df['salary_min'] = self.processed_df.get('salary_min_extracted')
        if 'salary_max' not in self.processed_df.columns:
            self.processed_df['salary_max'] = self.processed_df.get('salary_max_extracted')
    
    def _standardize_experience(self):
        """Standardize experience levels"""
        if self.processed_df is None:
            return
            
        exp_mapping = {
            'débutant': 'Entry Level',
            'junior': 'Entry Level', 
            'confirmé': 'Mid Level',
            'sénior': 'Senior Level',
            'expert': 'Expert Level',
            'manager': 'Management',
            'directeur': 'Executive'
        }
        
        if 'experience' in self.processed_df.columns:
            self.processed_df['experience_standardized'] = (
                self.processed_df['experience']
                .str.lower()
                .map(exp_mapping)
                .fillna(self.processed_df['experience'])
            )
    
    def _standardize_locations(self):
        """Standardize location names"""
        if self.processed_df is None:
            return
            
        location_mapping = {
            'casa': 'Casablanca',
            'casablanca': 'Casablanca',
            'rabat': 'Rabat',
            'fès': 'Fès',
            'fez': 'Fès',
            'marrakech': 'Marrakech',
            'marrakesh': 'Marrakech',
            'agadir': 'Agadir',
            'tanger': 'Tanger',
            'tangier': 'Tanger'
        }
        
        if 'location' in self.processed_df.columns:
            self.processed_df['location_standardized'] = (
                self.processed_df['location']
                .str.lower()
                .map(location_mapping)
                .fillna(self.processed_df['location'])
            )
    
    def _classify_sectors(self):
        """Advanced sector classification using keywords"""
        if self.processed_df is None:
            return
            
        sector_keywords = {
            'IT & Technology': ['informatique', 'technology', 'software', 'développement', 'programmation', 'it'],
            'Finance & Banking': ['banque', 'finance', 'comptabilité', 'banking', 'financial'],
            'Healthcare': ['santé', 'médical', 'health', 'healthcare', 'pharmaceutical'],
            'Education': ['éducation', 'enseignement', 'formation', 'education', 'teaching'],
            'Manufacturing': ['industrie', 'manufacturing', 'production', 'usine'],
            'Sales & Marketing': ['vente', 'marketing', 'commercial', 'sales', 'publicité'],
            'Engineering': ['ingénierie', 'engineering', 'technique', 'technical'],
            'Consulting': ['conseil', 'consulting', 'advisory', 'consultancy']
        }
        
        def classify_sector(description, title, current_sector):
            text = f"{description} {title}".lower()
            
            for sector, keywords in sector_keywords.items():
                if any(keyword in text for keyword in keywords):
                    return sector
            
            return current_sector if pd.notna(current_sector) else 'Other'
        
        if 'description_clean' in self.processed_df.columns:
            self.processed_df['sector_classified'] = self.processed_df.apply(
                lambda row: classify_sector(
                    row.get('description_clean', ''),
                    row.get('title_clean', ''),
                    row.get('sector', '')
                ), axis=1
            )
    
    def _extract_skills(self):
        """Extract technical skills from job descriptions"""
        if self.processed_df is None:
            return
            
        def extract_job_skills(description):
            if pd.isna(description):
                return []
            
            description = description.lower()
            found_skills = []
            
            for category, skills in self.tech_skills.items():
                for skill in skills:
                    if skill in description:
                        found_skills.append(skill)
            
            return found_skills
        
        if 'description_clean' in self.processed_df.columns:
            self.processed_df['extracted_skills'] = (
                self.processed_df['description_clean'].apply(extract_job_skills)
            )
            
            # Create skill frequency data
            all_skills = []
            for skills_list in self.processed_df['extracted_skills']:
                all_skills.extend(skills_list)
            
            from collections import Counter
            self.skills_frequency = Counter(all_skills)
    
    def perform_eda(self) -> Dict[str, Any]:
        """
        Comprehensive Exploratory Data Analysis
        Returns: Dictionary with all EDA results and visualizations
        """
        if self.processed_df is None:
            return {}
        
        print("🔍 Performing comprehensive EDA...")
        
        eda_results = {
            'basic_stats': self._get_basic_statistics(),
            'visualizations': {
                'distribution_plots': self._create_distribution_plots(),
                'correlation_analysis': self._perform_correlation_analysis(),
                'geographical_analysis': self._analyze_geographical_distribution(),
                'temporal_analysis': self._analyze_temporal_patterns(),
                'salary_analysis': self._analyze_salary_patterns(),
                'skills_analysis': self._analyze_skills_demand()
            },
            'data_quality': self._assess_data_quality(),
            'insights': self._generate_eda_insights()
        }
        
        return eda_results
    
    def _get_basic_statistics(self) -> Dict:
        """Generate comprehensive basic statistics"""
        if self.processed_df is None:
            return {}
            
        stats = {
            'dataset_info': {
                'total_jobs': len(self.processed_df),
                'unique_companies': self.processed_df['company'].nunique() if 'company' in self.processed_df.columns else 0,
                'unique_locations': self.processed_df['location'].nunique() if 'location' in self.processed_df.columns else 0,
                'unique_sectors': self.processed_df['sector'].nunique() if 'sector' in self.processed_df.columns else 0,
                'date_range': {
                    'earliest': self.processed_df['date_scraped'].min() if 'date_scraped' in self.processed_df.columns else None,
                    'latest': self.processed_df['date_scraped'].max() if 'date_scraped' in self.processed_df.columns else None
                }
            },
            'categorical_distributions': {},
            'numerical_stats': {},
            'missing_data': {}
        }
        
        # Categorical distributions
        categorical_cols = ['sector', 'location', 'experience', 'contract_type', 'source']
        for col in categorical_cols:
            if col in self.processed_df.columns:
                stats['categorical_distributions'][col] = (
                    self.processed_df[col].value_counts().head(10).to_dict()
                )
        
        # Numerical statistics
        numerical_cols = ['salary_min', 'salary_max', 'view_count', 'likes']
        for col in numerical_cols:
            if col in self.processed_df.columns:
                stats['numerical_stats'][col] = {
                    'mean': self.processed_df[col].mean(),
                    'median': self.processed_df[col].median(),
                    'std': self.processed_df[col].std(),
                    'min': self.processed_df[col].min(),
                    'max': self.processed_df[col].max()
                }
        
        # Missing data analysis
        for col in self.processed_df.columns:
            missing_count = self.processed_df[col].isnull().sum()
            missing_pct = (missing_count / len(self.processed_df)) * 100
            stats['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_pct
            }
        
        return stats
    
    def _create_distribution_plots(self) -> Dict[str, go.Figure]:
        """Create comprehensive distribution visualizations"""
        if self.processed_df is None:
            return {}
            
        plots = {}
        
        # 1. Sector distribution
        if 'sector_classified' in self.processed_df.columns:
            sector_counts = self.processed_df['sector_classified'].value_counts()
            plots['sector_distribution'] = px.treemap(
                names=sector_counts.index,
                values=sector_counts.values,
                title="Job Distribution by Sector",
                color=sector_counts.values,
                color_continuous_scale="viridis"
            )
        
        # 2. Location distribution
        if 'location_standardized' in self.processed_df.columns:
            location_counts = self.processed_df['location_standardized'].value_counts().head(15)
            plots['location_distribution'] = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="Top 15 Job Locations",
                color=location_counts.values,
                color_continuous_scale="plasma"
            )
        
        # 3. Experience level distribution
        if 'experience_standardized' in self.processed_df.columns:
            exp_counts = self.processed_df['experience_standardized'].value_counts()
            plots['experience_distribution'] = px.pie(
                values=exp_counts.values,
                names=exp_counts.index,
                title="Experience Level Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
        
        # 4. Contract type distribution
        if 'contract_type' in self.processed_df.columns:
            contract_counts = self.processed_df['contract_type'].value_counts()
            plots['contract_distribution'] = px.funnel(
                y=contract_counts.index,
                x=contract_counts.values,
                title="Contract Type Distribution"
            )
        
        # 5. Company hiring activity
        if 'company' in self.processed_df.columns:
            company_counts = self.processed_df['company'].value_counts().head(20)
            plots['top_companies'] = px.bar(
                x=company_counts.values,
                y=company_counts.index,
                orientation='h',
                title="Top 20 Companies by Job Postings",
                color=company_counts.values,
                color_continuous_scale="blues"
            )
        
        return plots
    
    def _perform_correlation_analysis(self) -> Dict[str, go.Figure]:
        """Perform correlation analysis on numerical variables"""
        if self.processed_df is None:
            return {}
            
        plots = {}
        
        # Select numerical columns
        numerical_cols = ['salary_min', 'salary_max', 'view_count', 'likes']
        available_cols = [col for col in numerical_cols if col in self.processed_df.columns]
        
        if len(available_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = self.processed_df[available_cols].corr()
            
            # Create heatmap
            plots['correlation_heatmap'] = px.imshow(
                corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                color_continuous_scale='RdBu',
                title="Correlation Matrix of Numerical Variables",
                aspect="auto"
            )
            
            # Scatter plot matrix
            plots['scatter_matrix'] = px.scatter_matrix(
                self.processed_df[available_cols].dropna(),
                title="Scatter Plot Matrix"
            )
        
        return plots
    
    def _analyze_geographical_distribution(self) -> Dict[str, go.Figure]:
        """Analyze geographical distribution of jobs"""
        plots = {}
        
        if 'location_standardized' in self.processed_df.columns:
            # Location vs Sector heatmap
            if 'sector_classified' in self.processed_df.columns:
                location_sector = pd.crosstab(
                    self.processed_df['location_standardized'],
                    self.processed_df['sector_classified']
                )
                
                plots['location_sector_heatmap'] = px.imshow(
                    location_sector.values,
                    x=location_sector.columns,
                    y=location_sector.index,
                    title="Job Distribution: Location vs Sector",
                    aspect="auto",
                    color_continuous_scale="viridis"
                )
            
            # Salary by location
            if 'salary_min' in self.processed_df.columns:
                location_salary = (
                    self.processed_df.groupby('location_standardized')['salary_min']
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )
                
                plots['salary_by_location'] = px.bar(
                    x=location_salary.values,
                    y=location_salary.index,
                    orientation='h',
                    title="Average Salary by Location (Top 10)",
                    color=location_salary.values,
                    color_continuous_scale="greens"
                )
        
        return plots
    
    def _analyze_temporal_patterns(self) -> Dict[str, go.Figure]:
        """Analyze temporal patterns in job postings"""
        plots = {}
        
        if 'date_scraped' in self.processed_df.columns:
            # Daily job postings trend
            daily_posts = (
                self.processed_df.groupby(self.processed_df['date_scraped'].dt.date)
                .size()
                .reset_index()
            )
            daily_posts.columns = ['date', 'count']
            
            plots['daily_trend'] = px.line(
                daily_posts,
                x='date',
                y='count',
                title="Daily Job Postings Trend",
                markers=True
            )
            
            # Weekly patterns
            self.processed_df['weekday'] = self.processed_df['date_scraped'].dt.day_name()
            weekday_counts = self.processed_df['weekday'].value_counts()
            
            plots['weekday_pattern'] = px.bar(
                x=weekday_counts.index,
                y=weekday_counts.values,
                title="Job Postings by Day of Week"
            )
            
            # Monthly trends by sector
            if 'sector_classified' in self.processed_df.columns:
                monthly_sector = (
                    self.processed_df.groupby([
                        self.processed_df['date_scraped'].dt.to_period('M'),
                        'sector_classified'
                    ]).size().reset_index(name='count')
                )
                monthly_sector['date_scraped'] = monthly_sector['date_scraped'].astype(str)
                
                plots['monthly_sector_trend'] = px.line(
                    monthly_sector,
                    x='date_scraped',
                    y='count',
                    color='sector_classified',
                    title="Monthly Job Postings by Sector"
                )
        
        return plots
    
    def _analyze_salary_patterns(self) -> Dict[str, go.Figure]:
        """Comprehensive salary analysis"""
        plots = {}
        
        if 'salary_min' in self.processed_df.columns:
            salary_df = self.processed_df.dropna(subset=['salary_min'])
            
            # Salary distribution
            plots['salary_distribution'] = px.histogram(
                salary_df,
                x='salary_min',
                nbins=30,
                title="Salary Distribution",
                marginal="box"
            )
            
            # Salary by sector
            if 'sector_classified' in salary_df.columns:
                plots['salary_by_sector'] = px.box(
                    salary_df,
                    x='sector_classified',
                    y='salary_min',
                    title="Salary Distribution by Sector"
                )
            
            # Salary by experience
            if 'experience_standardized' in salary_df.columns:
                plots['salary_by_experience'] = px.box(
                    salary_df,
                    x='experience_standardized',
                    y='salary_min',
                    title="Salary Distribution by Experience Level"
                )
            
            # Salary trends over time
            if 'date_scraped' in salary_df.columns:
                monthly_salary = (
                    salary_df.groupby(salary_df['date_scraped'].dt.to_period('M'))['salary_min']
                    .mean()
                    .reset_index()
                )
                monthly_salary['date_scraped'] = monthly_salary['date_scraped'].astype(str)
                
                plots['salary_trend'] = px.line(
                    monthly_salary,
                    x='date_scraped',
                    y='salary_min',
                    title="Average Salary Trend Over Time",
                    markers=True
                )
        
        return plots
    
    def _analyze_skills_demand(self) -> Dict[str, go.Figure]:
        """Analyze skills demand and trends"""
        plots = {}
        
        if hasattr(self, 'skills_frequency') and self.skills_frequency:
            # Top skills bar chart
            top_skills = dict(self.skills_frequency.most_common(20))
            
            plots['top_skills'] = px.bar(
                x=list(top_skills.values()),
                y=list(top_skills.keys()),
                orientation='h',
                title="Top 20 Most Demanded Skills",
                color=list(top_skills.values()),
                color_continuous_scale="viridis"
            )
            
            # Skills by category
            category_skills = {}
            for category, skills in self.tech_skills.items():
                category_count = sum(self.skills_frequency.get(skill, 0) for skill in skills)
                category_skills[category] = category_count
            
            plots['skills_by_category'] = px.pie(
                values=list(category_skills.values()),
                names=list(category_skills.keys()),
                title="Skills Demand by Category"
            )
            
            # Skills evolution over time
            if 'date_scraped' in self.processed_df.columns:
                # This is a simplified version - you might want to implement more sophisticated tracking
                plots['skills_trend_note'] = go.Figure().add_annotation(
                    text="Skills trend analysis requires historical data tracking",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
        
        return plots
    
    def _assess_data_quality(self) -> Dict:
        """Comprehensive data quality assessment"""
        quality_report = {
            'completeness': {},
            'consistency': {},
            'validity': {},
            'overall_score': 0
        }
        
        # Completeness analysis
        for col in self.processed_df.columns:
            non_null_pct = (self.processed_df[col].notna().sum() / len(self.processed_df)) * 100
            quality_report['completeness'][col] = non_null_pct
        
        # Consistency checks
        quality_report['consistency']['duplicate_jobs'] = self.processed_df.duplicated().sum()
        
        # Validity checks
        if 'salary_min' in self.processed_df.columns and 'salary_max' in self.processed_df.columns:
            invalid_salary = (
                self.processed_df['salary_min'] > self.processed_df['salary_max']
            ).sum()
            quality_report['validity']['invalid_salary_ranges'] = invalid_salary
        
        # Calculate overall score
        avg_completeness = np.mean(list(quality_report['completeness'].values()))
        consistency_score = 100 - (quality_report['consistency']['duplicate_jobs'] / len(self.processed_df) * 100)
        
        quality_report['overall_score'] = (avg_completeness + consistency_score) / 2
        
        return quality_report
    
    def _generate_eda_insights(self) -> List[str]:
        """Generate key insights from EDA"""
        insights = []
        
        if self.processed_df is None:
            return insights
        
        # Dataset size insight
        insights.append(f"📊 Dataset contains {len(self.processed_df):,} job postings")
        
        # Top sector insight
        if 'sector_classified' in self.processed_df.columns:
            top_sector = self.processed_df['sector_classified'].value_counts().index[0]
            top_sector_count = self.processed_df['sector_classified'].value_counts().iloc[0]
            insights.append(f"🏆 {top_sector} is the dominant sector with {top_sector_count:,} jobs")
        
        # Location insight
        if 'location_standardized' in self.processed_df.columns:
            top_location = self.processed_df['location_standardized'].value_counts().index[0]
            insights.append(f"📍 {top_location} has the highest concentration of job opportunities")
        
        # Salary insight
        if 'salary_min' in self.processed_df.columns:
            avg_salary = self.processed_df['salary_min'].mean()
            insights.append(f"💰 Average minimum salary is {avg_salary:,.0f} MAD")
        
        # Skills insight
        if hasattr(self, 'skills_frequency') and self.skills_frequency:
            top_skill = self.skills_frequency.most_common(1)[0]
            insights.append(f"🔧 Most demanded skill: {top_skill[0]} ({top_skill[1]} mentions)")
        
        return insights
    
    def analyze_emerging_trends(self) -> Dict[str, any]:
        """
        Analyze emerging trends and predict future market movements
        """
        print("📈 Analyzing emerging trends...")
        
        trends_analysis = {
            'emerging_sectors': self._identify_emerging_sectors(),
            'skill_trends': self._analyze_skill_trends(),
            'salary_trends': self._analyze_salary_trends(),
            'location_trends': self._analyze_location_trends(),
            'growth_predictions': self._predict_market_growth()
        }
        
        return trends_analysis
    
    def _identify_emerging_sectors(self) -> Dict:
        """Identify emerging and declining sectors"""
        if 'date_scraped' not in self.processed_df.columns or 'sector_classified' not in self.processed_df.columns:
            return {}
        
        # Calculate sector growth rates
        sector_monthly = (
            self.processed_df.groupby([
                self.processed_df['date_scraped'].dt.to_period('M'),
                'sector_classified'
            ]).size().reset_index(name='count')
        )
        
        growth_rates = {}
        for sector in self.processed_df['sector_classified'].unique():
            sector_data = sector_monthly[sector_monthly['sector_classified'] == sector]
            if len(sector_data) > 1:
                # Calculate simple growth rate
                first_month = sector_data['count'].iloc[0]
                last_month = sector_data['count'].iloc[-1]
                growth_rate = ((last_month - first_month) / first_month) * 100 if first_month > 0 else 0
                growth_rates[sector] = growth_rate
        
        # Sort by growth rate
        emerging = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)[:5]
        declining = sorted(growth_rates.items(), key=lambda x: x[1])[:5]
        
        return {
            'emerging': emerging,
            'declining': declining,
            'growth_rates': growth_rates
        }
    
    def _analyze_skill_trends(self) -> Dict:
        """Analyze trending skills"""
        if not hasattr(self, 'skills_frequency'):
            return {}
        
        # For now, return current demand - would need historical data for true trends
        trending_skills = dict(self.skills_frequency.most_common(10))
        
        return {
            'current_top_skills': trending_skills,
            'emerging_skills': list(trending_skills.keys())[:5],
            'skill_categories': {
                category: sum(self.skills_frequency.get(skill, 0) for skill in skills)
                for category, skills in self.tech_skills.items()
            }
        }
    
    def _analyze_salary_trends(self) -> Dict:
        """Analyze salary trends and projections"""
        if 'salary_min' not in self.processed_df.columns:
            return {}
        
        salary_trends = {
            'overall_stats': {
                'mean': self.processed_df['salary_min'].mean(),
                'median': self.processed_df['salary_min'].median(),
                'std': self.processed_df['salary_min'].std()
            },
            'by_sector': {},
            'by_experience': {},
            'by_location': {}
        }
        
        # Salary by sector
        if 'sector_classified' in self.processed_df.columns:
            salary_trends['by_sector'] = (
                self.processed_df.groupby('sector_classified')['salary_min']
                .agg(['mean', 'median', 'count'])
                .to_dict('index')
            )
        
        # Salary by experience
        if 'experience_standardized' in self.processed_df.columns:
            salary_trends['by_experience'] = (
                self.processed_df.groupby('experience_standardized')['salary_min']
                .agg(['mean', 'median', 'count'])
                .to_dict('index')
            )
        
        # Salary by location
        if 'location_standardized' in self.processed_df.columns:
            salary_trends['by_location'] = (
                self.processed_df.groupby('location_standardized')['salary_min']
                .agg(['mean', 'median', 'count'])
                .to_dict('index')
            )
        
        return salary_trends
    
    def _analyze_location_trends(self) -> Dict:
        """Analyze location-based trends"""
        if 'location_standardized' not in self.processed_df.columns:
            return {}
        
        location_stats = (
            self.processed_df.groupby('location_standardized')
            .agg({
                'title': 'count',
                'salary_min': 'mean' if 'salary_min' in self.processed_df.columns else lambda x: None,
                'sector_classified': lambda x: x.mode().iloc[0] if len(x) > 0 else None
            })
            .rename(columns={'title': 'job_count', 'salary_min': 'avg_salary', 'sector_classified': 'dominant_sector'})
        )
        
        return location_stats.to_dict('index')
    
    def _predict_market_growth(self) -> Dict:
        """Simple market growth predictions based on current trends"""
        predictions = {
            'methodology': 'Based on current data patterns and trends',
            'growth_sectors': [],
            'recommendations': []
        }
        
        # Get sector growth data
        emerging_data = self._identify_emerging_sectors()
        if emerging_data and 'emerging' in emerging_data:
            predictions['growth_sectors'] = [sector for sector, rate in emerging_data['emerging'][:3]]
        
        # Generate recommendations
        if hasattr(self, 'skills_frequency'):
            top_skills = list(dict(self.skills_frequency.most_common(5)).keys())
            predictions['recommendations'] = [
                f"Focus on developing skills in: {', '.join(top_skills)}",
                "Consider opportunities in emerging sectors",
                "Monitor salary trends for negotiation insights"
            ]
        
        return predictions
    
    def generate_streamlit_dashboard(self) -> None:
        """
        Generate comprehensive Streamlit dashboard for EDA results
        This method should be called from within a Streamlit app
        """
        if self.processed_df is None:
            st.error("No data available for analysis")
            return
        
        st.markdown("## 🔍 Advanced Job Market Analysis Dashboard")
        
        # Perform EDA
        eda_results = self.perform_eda()
        trends_analysis = self.analyze_emerging_trends()
        
        # Create tabs for different analysis sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "📈 Distributions", "🔗 Correlations", 
            "🌍 Geographic", "📅 Temporal"
        ])
        
        with tab1:
            self._display_overview_tab(eda_results)
        
        with tab2:
            self._display_distributions_tab(eda_results)
        
        with tab3:
            self._display_correlations_tab(eda_results)
        
        with tab4:
            self._display_geographic_tab(eda_results)
        
        with tab5:
            self._display_temporal_tab(eda_results)
        
        # Additional analysis sections
        st.markdown("---")
        
        # Skills Analysis
        st.markdown("## 🔧 Skills & Competencies Analysis")
        self._display_skills_analysis(eda_results, trends_analysis)
        
        # Emerging Trends
        st.markdown("## 📈 Emerging Trends & Predictions")
        self._display_trends_analysis(trends_analysis)
        
        # Data Quality Report
        st.markdown("## 🔍 Data Quality Assessment")
        self._display_data_quality(eda_results)
    
    def _display_overview_tab(self, eda_results):
        """Display overview tab content"""
        basic_stats = eda_results.get('basic_stats', {})
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        dataset_info = basic_stats.get('dataset_info', {})
        with col1:
            st.metric("Total Jobs", f"{dataset_info.get('total_jobs', 0):,}")
        with col2:
            st.metric("Companies", f"{dataset_info.get('unique_companies', 0):,}")
        with col3:
            st.metric("Locations", f"{dataset_info.get('unique_locations', 0):,}")
        with col4:
            st.metric("Sectors", f"{dataset_info.get('unique_sectors', 0):,}")
        
        # Key insights
        insights = eda_results.get('insights', [])
        if insights:
            st.markdown("### 🔍 Key Insights")
            for insight in insights:
                st.write(f"• {insight}")
        
        # Data quality score
        data_quality = eda_results.get('data_quality', {})
        if data_quality:
            quality_score = data_quality.get('overall_score', 0)
            st.metric("Data Quality Score", f"{quality_score:.1f}%")
    
    def _display_distributions_tab(self, eda_results):
        """Display distributions tab content"""
        plots = eda_results.get('visualizations', {}).get('distribution_plots', {})
        
        for plot_name, fig in plots.items():
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_correlations_tab(self, eda_results):
        """Display correlations tab content"""
        plots = eda_results.get('visualizations', {}).get('correlation_analysis', {})
        
        for plot_name, fig in plots.items():
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_geographic_tab(self, eda_results):
        """Display geographic tab content"""
        plots = eda_results.get('visualizations', {}).get('geographical_analysis', {})
        
        for plot_name, fig in plots.items():
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_temporal_tab(self, eda_results):
        """Display temporal tab content"""
        plots = eda_results.get('visualizations', {}).get('temporal_analysis', {})
        
        for plot_name, fig in plots.items():
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    def _display_skills_analysis(self, eda_results, trends_analysis):
        """Display skills analysis section"""
        skills_plots = eda_results.get('visualizations', {}).get('skills_analysis', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'top_skills' in skills_plots:
                st.plotly_chart(skills_plots['top_skills'], use_container_width=True)
        
        with col2:
            if 'skills_by_category' in skills_plots:
                st.plotly_chart(skills_plots['skills_by_category'], use_container_width=True)
        
        # Skills trends data
        skill_trends = trends_analysis.get('skill_trends', {})
        if skill_trends:
            st.markdown("### 📈 Skills Trends")
            
            emerging_skills = skill_trends.get('emerging_skills', [])
            if emerging_skills:
                st.write("**🔥 Emerging Skills:**")
                st.write(", ".join(emerging_skills))
    
    def _display_trends_analysis(self, trends_analysis):
        """Display trends analysis section"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 Emerging Sectors")
            emerging_sectors = trends_analysis.get('emerging_sectors', {})
            if emerging_sectors and 'emerging' in emerging_sectors:
                for sector, growth_rate in emerging_sectors['emerging']:
                    st.write(f"• **{sector}**: {growth_rate:.1f}% growth")
        
        with col2:
            st.markdown("### 📉 Declining Sectors")
            if emerging_sectors and 'declining' in emerging_sectors:
                for sector, growth_rate in emerging_sectors['declining']:
                    st.write(f"• **{sector}**: {growth_rate:.1f}% change")
        
        # Growth predictions
        predictions = trends_analysis.get('growth_predictions', {})
        if predictions:
            st.markdown("### 🔮 Market Predictions")
            recommendations = predictions.get('recommendations', [])
            for rec in recommendations:
                st.write(f"• {rec}")
    
    def _display_data_quality(self, eda_results):
        """Display data quality assessment"""
        quality = eda_results.get('data_quality', {})
        
        if quality:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Completeness Score")
                completeness = quality.get('completeness', {})
                for field, score in completeness.items():
                    if score < 70:
                        color = "🔴"
                    elif score < 90:
                        color = "🟡"
                    else:
                        color = "🟢"
                    st.write(f"{color} {field}: {score:.1f}%")
            
            with col2:
                st.markdown("### ⚠️ Quality Issues")
                consistency = quality.get('consistency', {})
                validity = quality.get('validity', {})
                
                duplicates = consistency.get('duplicate_jobs', 0)
                if duplicates > 0:
                    st.write(f"• {duplicates} duplicate job postings found")
                
                invalid_salaries = validity.get('invalid_salary_ranges', 0)
                if invalid_salaries > 0:
                    st.write(f"• {invalid_salaries} invalid salary ranges")
    
    def export_analysis_report(self, filename: str = None) -> str:
        """
        Export comprehensive analysis report to file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"job_market_analysis_{timestamp}.json"
        
        # Perform full analysis
        eda_results = self.perform_eda()
        trends_analysis = self.analyze_emerging_trends()
        
        # Combine all results
        full_report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'dataset_size': len(self.processed_df) if self.processed_df is not None else 0,
                'analysis_version': '1.0'
            },
            'eda_results': eda_results,
            'trends_analysis': trends_analysis
        }
        
        # Save to file
        import json
        with open(filename, 'w', encoding='utf-8') as f:
            # Convert numpy types to native Python types for JSON serialization
            json.dump(full_report, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"✅ Analysis report exported to {filename}")
        return filename
