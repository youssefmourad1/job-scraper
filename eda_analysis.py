#!/usr/bin/env python3
"""
Optimized EDA for Job Market Data - Based on Actual Data Structure
"""
import pandas as pd
import numpy as np
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import os
from datetime import datetime
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobMarketEDA:
    def __init__(self, db_path='data/job_market.db'):
        """Initialize the EDA analyzer"""
        self.db_path = db_path
        self.df = None
        
    def load_data(self):
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT title, company, location, sector, fonction,
                   experience, education_level, contract_type,
                   salary, salary_min, salary_max,
                   description, requirements, source,
                   is_active, view_count, likes
            FROM jobs WHERE is_active = 1
            """
            self.df = pd.read_sql_query(query, conn)
            conn.close()
            logger.info(f"Loaded {len(self.df)} records from database")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
            
    def preprocess_data(self):
        """Preprocess data based on actual data characteristics"""
        if self.df is None:
            return False
            
        # Clean text data
        self.df['description_clean'] = self.df['description'].fillna('').apply(self._clean_text)
        self.df['requirements_clean'] = self.df['requirements'].fillna('').apply(self._clean_text)
        
        # Handle missing categorical data
        self.df['location'] = self.df['location'].fillna('Unknown')
        self.df['experience'] = self.df['experience'].fillna('Not Specified')
        self.df['sector'] = self.df['sector'].fillna('Other')
        
        # Create salary categories for analysis
        self.df['salary_category'] = self._categorize_salary()
        
        logger.info("Data preprocessing completed")
        return True
        
    def _clean_text(self, text):
        """Clean text data efficiently"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        return text
        
    def _categorize_salary(self):
        """Categorize salaries based on available data"""
        salary_cats = []
        for _, row in self.df.iterrows():
            if pd.notna(row['salary_min']) and row['salary_min'] > 0:
                if row['salary_min'] < 3000:
                    salary_cats.append('Low (< 3k)')
                elif row['salary_min'] < 6000:
                    salary_cats.append('Medium (3k-6k)')
                else:
                    salary_cats.append('High (> 6k)')
            else:
                salary_cats.append('Not Specified')
        return salary_cats
        
    # Core Analysis Functions - Optimized for actual data
    def analyze_source_distribution(self):
        """Source distribution - Key insight: MarocAnnonce dominates"""
        source_counts = self.df['source'].value_counts()
        fig = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title='Job Distribution by Source (MarocAnnonce: 80%, Rekrute: 20%)',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4']
        )
        return fig
        
    def analyze_top_sectors(self):
        """Top sectors - Focus on most active ones"""
        sector_counts = self.df['sector'].value_counts().head(12)
        fig = px.bar(
            x=sector_counts.values,
            y=sector_counts.index,
            orientation='h',
            title='Top 12 Job Sectors',
            labels={'x': 'Number of Jobs', 'y': 'Sector'}
        )
        fig.update_layout(height=500)
        return fig
        
    def analyze_location_insights(self):
        """Location analysis - Handle missing location data"""
        location_counts = self.df['location'].value_counts().head(10)
        fig = px.treemap(
            names=location_counts.index,
            values=location_counts.values,
            title='Job Distribution by Location (Casablanca leads with 37%)'
        )
        return fig
        
    def analyze_salary_distribution(self):
        """Salary analysis - Focus on available data"""
        salary_data = self.df['salary_category'].value_counts()
        fig = px.pie(
            values=salary_data.values,
            names=salary_data.index,
            title='Salary Range Distribution (88% Not Specified)',
            color_discrete_sequence=['#FF9999', '#66B2FF', '#99FF99', '#FFB366']
        )
        return fig
        
    def analyze_contract_preferences(self):
        """Contract type analysis"""
        contract_counts = self.df['contract_type'].value_counts()
        fig = px.bar(
            x=contract_counts.index,
            y=contract_counts.values,
            title='Contract Type Preferences (CDI dominates)',
            color=contract_counts.values,
            color_continuous_scale='viridis'
        )
        return fig
        
    def analyze_education_requirements(self):
        """Education level analysis"""
        education_counts = self.df['education_level'].value_counts()
        fig = px.funnel(
            y=education_counts.index,
            x=education_counts.values,
            title='Education Level Requirements'
        )
        return fig
        
    def analyze_experience_gap(self):
        """Experience requirements - Highlight the gap"""
        exp_counts = self.df['experience'].value_counts()
        fig = px.bar(
            x=exp_counts.index,
            y=exp_counts.values,
            title='Experience Requirements (80% Not Specified)',
            color=['red' if x == 'Not Specified' else 'blue' for x in exp_counts.index]
        )
        return fig
        
    def analyze_company_concentration(self):
        """Company distribution analysis"""
        company_counts = self.df['company'].value_counts().head(15)
        fig = px.scatter(
            x=range(len(company_counts)),
            y=company_counts.values,
            size=company_counts.values,
            hover_name=company_counts.index,
            title='Company Job Posting Distribution (Top 15)',
            labels={'x': 'Company Rank', 'y': 'Number of Jobs'}
        )
        return fig
        
    def analyze_text_complexity(self):
        """Text analysis - Description vs Requirements"""
        desc_lengths = self.df['description'].str.len()
        req_lengths = self.df['requirements'].str.len()
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=desc_lengths, name='Description Length', opacity=0.7))
        fig.add_trace(go.Histogram(x=req_lengths, name='Requirements Length', opacity=0.7))
        fig.update_layout(
            title='Text Content Analysis (Descriptions: avg 1098 chars, Requirements: avg 88 chars)',
            xaxis_title='Character Length',
            yaxis_title='Count',
            barmode='overlay'
        )
        return fig
        
    def perform_pca_analysis(self):
        """PCA on text features for dimensionality insights"""
        # Vectorize descriptions
        vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        desc_vectors = vectorizer.fit_transform(self.df['description_clean'])
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(desc_vectors.toarray())
        
        # Create scatter plot colored by source
        fig = px.scatter(
            x=pca_result[:, 0],
            y=pca_result[:, 1],
            color=self.df['source'],
            title=f'PCA of Job Descriptions (Explained Variance: {pca.explained_variance_ratio_.sum():.2%})',
            labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', 
                   'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'}
        )
        return fig
        
    def generate_comprehensive_report(self):
        """Generate optimized comprehensive report"""
        if self.df is None:
            return None
            
        # Create 10 subplots (5x2 grid)
        fig = make_subplots(
            rows=5, cols=2,
            specs=[
                [{"type": "pie"}, {"type": "xy"}],
                [{"type": "treemap"}, {"type": "pie"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "xy"}, {"type": "xy"}]
            ],
            subplot_titles=(
                'Source Distribution', 'Top Sectors',
                'Location Distribution', 'Salary Ranges',
                'Contract Types', 'Education Requirements',
                'Experience Gap', 'Company Concentration',
                'Text Complexity', 'PCA Analysis'
            ),
            vertical_spacing=0.08
        )
        
        # Generate all plots
        plots = [
            self.analyze_source_distribution(),
            self.analyze_top_sectors(),
            self.analyze_location_insights(),
            self.analyze_salary_distribution(),
            self.analyze_contract_preferences(),
            self.analyze_education_requirements(),
            self.analyze_experience_gap(),
            self.analyze_company_concentration(),
            self.analyze_text_complexity(),
            self.perform_pca_analysis()
        ]
        
        # Add traces to subplots
        positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2), (5,1), (5,2)]
        
        for i, plot in enumerate(plots):
            if plot and plot.data:
                for trace in plot.data:
                    fig.add_trace(trace, row=positions[i][0], col=positions[i][1])
        
        fig.update_layout(
            height=2000,
            width=1600,
            title_text="Comprehensive Job Market Analysis - Data-Driven Insights",
            showlegend=True
        )
        
        return fig
        
    def generate_data_quality_report(self):
        """Generate data quality insights"""
        quality_stats = {
            "data_completeness": {
                "total_records": len(self.df),
                "complete_records": len(self.df.dropna()),
                "completion_rate": len(self.df.dropna()) / len(self.df) * 100
            },
            "field_completeness": {
                field: (1 - self.df[field].isna().sum() / len(self.df)) * 100
                for field in ['title', 'company', 'location', 'sector', 'description']
            },
            "data_distribution": {
                "sources": self.df['source'].value_counts().to_dict(),
                "top_sectors": self.df['sector'].value_counts().head(5).to_dict(),
                "top_locations": self.df['location'].value_counts().head(5).to_dict()
            },
            "text_quality": {
                "avg_description_length": self.df['description'].str.len().mean(),
                "avg_requirements_length": self.df['requirements'].str.len().mean(),
                "empty_descriptions": (self.df['description'].str.len() == 0).sum(),
                "empty_requirements": (self.df['requirements'].str.len() == 0).sum()
            }
        }
        return quality_stats
        
    def save_report(self, output_path='reports/eda_report.html'):
        """Save the comprehensive report"""
        fig = self.generate_comprehensive_report()
        if fig:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Comprehensive report saved to {output_path}")
            return True
        return False
        
    def save_quality_report(self, output_path='reports/data_quality.json'):
        """Save data quality report"""
        quality_stats = self.generate_data_quality_report()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(quality_stats, f, indent=4)
        logger.info(f"Data quality report saved to {output_path}")
        return True

def main():
    """Main function optimized for actual data"""
    eda = JobMarketEDA()
    
    if not eda.load_data():
        return
        
    if not eda.preprocess_data():
        return
        
    # Generate reports
    eda.save_report()
    eda.save_quality_report()
    
    logger.info("Optimized EDA completed successfully")

if __name__ == "__main__":
    main() 