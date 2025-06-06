"""
Advanced EDA Engine for Job Market Analysis
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

from config.settings import ANALYTICS_CONFIG
from utils.helpers import calculate_growth_rate, format_number

logger = logging.getLogger('analytics.eda_engine')

class EDAEngine:
    """
    Comprehensive Exploratory Data Analysis Engine
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.analysis_results = {}
        self.figures = {}
        logger.info(f"Initialized EDA Engine with {len(df)} records")
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate a comprehensive EDA report"""
        logger.info("Generating comprehensive EDA report")
        
        report = {
            'data_overview': self.analyze_data_overview(),
            'distribution_analysis': self.analyze_distributions(),
            'correlation_analysis': self.analyze_correlations(),
            'trend_analysis': self.analyze_trends(),
            'skills_analysis': self.analyze_skills(),
            'salary_analysis': self.analyze_salaries(),
            'geographic_analysis': self.analyze_geography(),
            'temporal_analysis': self.analyze_temporal_patterns(),
            'advanced_insights': self.generate_advanced_insights()
        }
        
        logger.info("Comprehensive EDA report generated successfully")
        return report
    
    def analyze_data_overview(self) -> Dict:
        """Analyze basic data characteristics"""
        overview = {
            'total_records': len(self.df),
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicate_records': self.df.duplicated().sum(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Date range analysis
        if 'date_posted' in self.df.columns:
            overview['date_range'] = {
                'start': self.df['date_posted'].min(),
                'end': self.df['date_posted'].max(),
                'span_days': (self.df['date_posted'].max() - self.df['date_posted'].min()).days
            }
        
        # Categorical field analysis
        categorical_fields = ['source', 'sector', 'location', 'fonction', 'experience', 'contract_type']
        overview['categorical_summary'] = {}
        
        for field in categorical_fields:
            if field in self.df.columns:
                overview['categorical_summary'][field] = {
                    'unique_count': self.df[field].nunique(),
                    'most_common': self.df[field].value_counts().head(5).to_dict()
                }
        
        return overview
    
    def analyze_distributions(self) -> Dict:
        """Analyze distribution patterns"""
        distributions = {}
        
        # Categorical distributions
        categorical_fields = ['source', 'sector', 'location', 'fonction', 'experience_category']
        
        for field in categorical_fields:
            if field in self.df.columns:
                value_counts = self.df[field].value_counts()
                distributions[field] = {
                    'counts': value_counts.to_dict(),
                    'percentages': (value_counts / len(self.df) * 100).to_dict(),
                    'entropy': stats.entropy(value_counts.values),  # Measure of diversity
                    'concentration_ratio': value_counts.head(3).sum() / value_counts.sum()  # Top 3 concentration
                }
        
        # Numerical distributions
        numerical_fields = ['salary_avg', 'view_count', 'likes', 'skills_count']
        
        for field in numerical_fields:
            if field in self.df.columns and self.df[field].notna().sum() > 0:
                data = self.df[field].dropna()
                distributions[field] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max(),
                    'skewness': stats.skew(data),
                    'kurtosis': stats.kurtosis(data),
                    'percentiles': {
                        '25th': data.quantile(0.25),
                        '75th': data.quantile(0.75),
                        '90th': data.quantile(0.90),
                        '95th': data.quantile(0.95)
                    }
                }
        
        return distributions
    
    def analyze_correlations(self) -> Dict:
        """Analyze correlations between variables"""
        correlations = {}
        
        # Select numerical columns for correlation analysis
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 1:
            corr_matrix = self.df[numerical_cols].corr()
            
            # Find strong correlations (> 0.5 or < -0.5)
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                        })
            
            correlations = {
                'correlation_matrix': corr_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'highest_correlation': max(strong_correlations, 
                                         key=lambda x: abs(x['correlation']), 
                                         default=None)
            }
        
        return correlations
    
    def analyze_trends(self) -> Dict:
        """Analyze temporal trends and patterns"""
        trends = {}
        
        if 'date_posted' in self.df.columns:
            # Daily trends
            daily_counts = self.df.groupby(self.df['date_posted'].dt.date).size()
            trends['daily_stats'] = {
                'mean_daily_posts': daily_counts.mean(),
                'max_daily_posts': daily_counts.max(),
                'min_daily_posts': daily_counts.min(),
                'trend_direction': 'increasing' if daily_counts.tail(7).mean() > daily_counts.head(7).mean() else 'decreasing'
            }
            
            # Weekly patterns
            weekly_pattern = self.df.groupby(self.df['date_posted'].dt.dayofweek).size()
            trends['weekly_pattern'] = {
                'peak_day': weekly_pattern.idxmax(),
                'low_day': weekly_pattern.idxmin(),
                'weekend_vs_weekday': {
                    'weekend_avg': weekly_pattern[[5, 6]].mean(),
                    'weekday_avg': weekly_pattern[[0, 1, 2, 3, 4]].mean()
                }
            }
            
            # Growth analysis by sector
            if 'sector' in self.df.columns:
                sector_growth = {}
                for sector in self.df['sector'].value_counts().head(10).index:
                    sector_data = self.df[self.df['sector'] == sector]
                    if len(sector_data) > 10:  # Enough data for trend analysis
                        recent_count = len(sector_data[sector_data['date_posted'] >= 
                                                     datetime.now() - timedelta(days=7)])
                        previous_count = len(sector_data[
                            (sector_data['date_posted'] >= datetime.now() - timedelta(days=14)) &
                            (sector_data['date_posted'] < datetime.now() - timedelta(days=7))
                        ])
                        
                        if previous_count > 0:
                            growth_rate = calculate_growth_rate(recent_count, previous_count)
                            sector_growth[sector] = growth_rate
                
                trends['sector_growth'] = dict(sorted(sector_growth.items(), 
                                                    key=lambda x: x[1], reverse=True))
        
        return trends
    
    def analyze_skills(self) -> Dict:
        """Analyze skills demand and patterns"""
        skills_analysis = {}
        
        if 'extracted_skills' in self.df.columns:
            # Flatten all skills
            all_skills = []
            for skills_list in self.df['extracted_skills'].dropna():
                if isinstance(skills_list, list):
                    all_skills.extend(skills_list)
            
            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts()
                
                skills_analysis = {
                    'total_unique_skills': len(skill_counts),
                    'most_demanded': skill_counts.head(20).to_dict(),
                    'skill_diversity': len(skill_counts) / len(all_skills),  # Simpson's diversity
                    'average_skills_per_job': self.df['skills_count'].mean() if 'skills_count' in self.df.columns else 0
                }
                
                # Skills by sector analysis
                if 'sector' in self.df.columns:
                    sector_skills = {}
                    for sector in self.df['sector'].value_counts().head(5).index:
                        sector_data = self.df[self.df['sector'] == sector]
                        sector_all_skills = []
                        for skills_list in sector_data['extracted_skills'].dropna():
                            if isinstance(skills_list, list):
                                sector_all_skills.extend(skills_list)
                        
                        if sector_all_skills:
                            sector_skill_counts = pd.Series(sector_all_skills).value_counts()
                            sector_skills[sector] = sector_skill_counts.head(10).to_dict()
                    
                    skills_analysis['skills_by_sector'] = sector_skills
        
        return skills_analysis
    
    def analyze_salaries(self) -> Dict:
        """Analyze salary patterns and distributions"""
        salary_analysis = {}
        
        if 'salary_avg' in self.df.columns:
            salary_data = self.df.dropna(subset=['salary_avg'])
            
            if not salary_data.empty:
                salary_analysis = {
                    'overall_stats': {
                        'mean': salary_data['salary_avg'].mean(),
                        'median': salary_data['salary_avg'].median(),
                        'std': salary_data['salary_avg'].std(),
                        'min': salary_data['salary_avg'].min(),
                        'max': salary_data['salary_avg'].max()
                    }
                }
                
                # Salary by experience
                if 'experience_category' in salary_data.columns:
                    exp_salary = salary_data.groupby('experience_category')['salary_avg'].agg([
                        'mean', 'median', 'count'
                    ]).to_dict()
                    salary_analysis['by_experience'] = exp_salary
                
                # Salary by sector
                if 'sector' in salary_data.columns:
                    sector_salary = salary_data.groupby('sector')['salary_avg'].agg([
                        'mean', 'median', 'count'
                    ]).sort_values('mean', ascending=False).head(10).to_dict()
                    salary_analysis['by_sector'] = sector_salary
                
                # Salary by location
                if 'location' in salary_data.columns:
                    location_salary = salary_data.groupby('location')['salary_avg'].agg([
                        'mean', 'median', 'count'
                    ]).sort_values('mean', ascending=False).head(10).to_dict()
                    salary_analysis['by_location'] = location_salary
        
        return salary_analysis
    
    def analyze_geography(self) -> Dict:
        """Analyze geographic distribution and patterns"""
        geographic_analysis = {}
        
        if 'location' in self.df.columns:
            location_counts = self.df['location'].value_counts()
            
            # Major cities analysis
            major_cities = ['Casablanca', 'Rabat', 'Marrakech', 'Fès', 'Tanger']
            city_data = {}
            
            for city in major_cities:
                city_jobs = self.df[self.df['location'].str.contains(city, case=False, na=False)]
                if len(city_jobs) > 0:
                    city_data[city] = {
                        'job_count': len(city_jobs),
                        'percentage': len(city_jobs) / len(self.df) * 100,
                        'top_sectors': city_jobs['sector'].value_counts().head(5).to_dict() if 'sector' in city_jobs.columns else {},
                        'avg_salary': city_jobs['salary_avg'].mean() if 'salary_avg' in city_jobs.columns else None
                    }
            
            geographic_analysis = {
                'total_locations': len(location_counts),
                'top_locations': location_counts.head(15).to_dict(),
                'major_cities': city_data,
                'geographic_concentration': location_counts.head(5).sum() / location_counts.sum()  # Top 5 concentration
            }
        
        return geographic_analysis
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal posting patterns"""
        temporal_analysis = {}
        
        if 'date_posted' in self.df.columns:
            # Hour-of-day analysis (if time data available)
            if 'posting_weekday' in self.df.columns:
                weekday_pattern = self.df.groupby('posting_weekday').size()
                temporal_analysis['weekday_pattern'] = {
                    'counts': weekday_pattern.to_dict(),
                    'peak_weekday': weekday_pattern.idxmax(),
                    'lowest_weekday': weekday_pattern.idxmin()
                }
            
            # Monthly seasonality
            if 'posting_month' in self.df.columns:
                monthly_pattern = self.df.groupby('posting_month').size()
                temporal_analysis['monthly_pattern'] = {
                    'counts': monthly_pattern.to_dict(),
                    'peak_month': monthly_pattern.idxmax(),
                    'lowest_month': monthly_pattern.idxmin()
                }
            
            # Job posting frequency analysis
            posting_frequency = self.df.groupby(self.df['date_posted'].dt.date).size()
            temporal_analysis['posting_frequency'] = {
                'daily_average': posting_frequency.mean(),
                'daily_std': posting_frequency.std(),
                'most_active_day': posting_frequency.idxmax(),
                'least_active_day': posting_frequency.idxmin()
            }
        
        return temporal_analysis
    
    def generate_advanced_insights(self) -> Dict:
        """Generate advanced insights and recommendations"""
        insights = {
            'market_maturity': self._assess_market_maturity(),
            'competition_level': self._assess_competition_level(),
            'emerging_trends': self._identify_emerging_trends(),
            'recommendations': self._generate_recommendations()
        }
        
        return insights
    
    def _assess_market_maturity(self) -> Dict:
        """Assess job market maturity"""
        maturity_indicators = {}
        
        # Diversity metrics
        if 'sector' in self.df.columns:
            sector_diversity = self.df['sector'].nunique() / len(self.df)
            maturity_indicators['sector_diversity'] = sector_diversity
        
        if 'fonction' in self.df.columns:
            function_diversity = self.df['fonction'].nunique() / len(self.df)
            maturity_indicators['function_diversity'] = function_diversity
        
        # Skill complexity
        if 'skills_count' in self.df.columns:
            avg_skills = self.df['skills_count'].mean()
            maturity_indicators['average_skills_per_job'] = avg_skills
            maturity_indicators['skill_complexity'] = 'high' if avg_skills > 5 else 'medium' if avg_skills > 3 else 'low'
        
        return maturity_indicators
    
    def _assess_competition_level(self) -> Dict:
        """Assess competition level in the job market"""
        competition_metrics = {}
        
        # Job concentration
        if 'company' in self.df.columns:
            company_concentration = self.df['company'].value_counts().head(10).sum() / len(self.df)
            competition_metrics['company_concentration'] = company_concentration
            competition_metrics['market_concentration'] = 'high' if company_concentration > 0.5 else 'medium' if company_concentration > 0.3 else 'low'
        
        # Position availability
        if 'fonction' in self.df.columns:
            function_availability = self.df['fonction'].value_counts()
            competition_metrics['most_competitive_roles'] = function_availability.tail(10).to_dict()
            competition_metrics['least_competitive_roles'] = function_availability.head(10).to_dict()
        
        return competition_metrics
    
    def _identify_emerging_trends(self) -> Dict:
        """Identify emerging trends in the job market"""
        trends = {}
        
        # Recent growth in sectors
        if 'date_posted' in self.df.columns and 'sector' in self.df.columns:
            recent_date = self.df['date_posted'].max() - timedelta(days=30)
            recent_jobs = self.df[self.df['date_posted'] >= recent_date]
            
            if len(recent_jobs) > 0:
                recent_sector_dist = recent_jobs['sector'].value_counts(normalize=True)
                overall_sector_dist = self.df['sector'].value_counts(normalize=True)
                
                emerging_sectors = {}
                for sector in recent_sector_dist.index:
                    if sector in overall_sector_dist.index:
                        growth = (recent_sector_dist[sector] - overall_sector_dist[sector]) / overall_sector_dist[sector]
                        if growth > 0.2:  # 20% growth
                            emerging_sectors[sector] = growth
                
                trends['emerging_sectors'] = dict(sorted(emerging_sectors.items(), 
                                                       key=lambda x: x[1], reverse=True))
        
        return trends
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        missing_data = self.df.isnull().sum()
        if missing_data.max() > len(self.df) * 0.3:
            recommendations.append("Improve data collection to reduce missing values (>30% missing in some fields)")
        
        # Market insights recommendations
        if 'salary_avg' in self.df.columns:
            salary_data = self.df.dropna(subset=['salary_avg'])
            if len(salary_data) / len(self.df) < 0.3:
                recommendations.append("Enhance salary data collection for better market insights")
        
        # Skills analysis recommendations
        if 'extracted_skills' in self.df.columns:
            skills_coverage = self.df['extracted_skills'].notna().sum() / len(self.df)
            if skills_coverage < 0.5:
                recommendations.append("Improve skill extraction algorithms to capture more job requirements")
        
        # Geographic coverage recommendations
        if 'location' in self.df.columns:
            location_concentration = self.df['location'].value_counts().head(3).sum() / len(self.df)
            if location_concentration > 0.8:
                recommendations.append("Expand geographic coverage beyond major cities")
        
        return recommendations 