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
from typing import List, Dict, Tuple, Optional
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
