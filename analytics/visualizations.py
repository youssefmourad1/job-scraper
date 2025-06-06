"""
Visualization Engine for Job Market Analysis
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import Dict, List, Optional
import logging

logger = logging.getLogger('analytics.visualizations')

class VisualizationEngine:
    """
    Professional visualization engine for job market data
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.color_palette = px.colors.qualitative.Set3
        logger.info(f"Initialized Visualization Engine with {len(df)} records")
    
    def create_distribution_charts(self) -> Dict[str, go.Figure]:
        """Create distribution charts for categorical variables"""
        figures = {}
        
        # Source distribution
        if 'source' in self.df.columns:
            source_counts = self.df['source'].value_counts()
            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Job Distribution by Source",
                color_discrete_sequence=self.color_palette
            )
            figures['source_distribution'] = fig
        
        # Location distribution
        if 'location' in self.df.columns:
            location_counts = self.df['location'].value_counts().head(10)
            fig = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="Top 10 Job Locations",
                color=location_counts.values,
                color_continuous_scale="viridis"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            figures['location_distribution'] = fig
        
        # Sector distribution
        if 'sector' in self.df.columns:
            sector_counts = self.df['sector'].value_counts().head(15)
            fig = px.treemap(
                names=sector_counts.index,
                values=sector_counts.values,
                title="Job Distribution by Sector"
            )
            figures['sector_distribution'] = fig
        
        return figures
    
    def create_trend_charts(self) -> Dict[str, go.Figure]:
        """Create trend analysis charts"""
        figures = {}
        
        if 'date_posted' in self.df.columns:
            # Daily job postings trend
            daily_posts = self.df.groupby(self.df['date_posted'].dt.date).size()
            fig = px.line(
                x=daily_posts.index,
                y=daily_posts.values,
                title="Daily Job Postings Trend",
                labels={'x': 'Date', 'y': 'Number of Posts'}
            )
            figures['daily_trend'] = fig
            
            # Weekly trends by sector
            if 'sector' in self.df.columns:
                self.df['week'] = self.df['date_posted'].dt.to_period('W')
                weekly_sector = self.df.groupby(['week', 'sector']).size().reset_index(name='count')
                weekly_sector['week'] = weekly_sector['week'].astype(str)
                
                # Filter to top 5 sectors
                top_sectors = self.df['sector'].value_counts().head(5).index
                weekly_sector_filtered = weekly_sector[weekly_sector['sector'].isin(top_sectors)]
                
                fig = px.line(
                    weekly_sector_filtered,
                    x='week',
                    y='count',
                    color='sector',
                    title="Weekly Job Postings by Top Sectors",
                    labels={'week': 'Week', 'count': 'Number of Posts'}
                )
                figures['weekly_sector_trend'] = fig
        
        return figures
    
    def create_skills_charts(self) -> Dict[str, go.Figure]:
        """Create skills analysis charts"""
        figures = {}
        
        if 'extracted_skills' in self.df.columns:
            # Skills demand chart
            all_skills = []
            for skills_list in self.df['extracted_skills'].dropna():
                if isinstance(skills_list, list):
                    all_skills.extend(skills_list)
            
            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts().head(20)
                
                fig = px.bar(
                    x=skill_counts.values,
                    y=skill_counts.index,
                    orientation='h',
                    title="Top 20 Most Demanded Skills",
                    color=skill_counts.values,
                    color_continuous_scale="plasma"
                )
                fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
                figures['skills_demand'] = fig
        
        return figures
    
    def create_salary_charts(self) -> Dict[str, go.Figure]:
        """Create salary analysis charts"""
        figures = {}
        
        if 'salary_avg' in self.df.columns and self.df['salary_avg'].notna().sum() > 0:
            # Salary distribution
            salary_data = self.df.dropna(subset=['salary_avg'])
            fig = px.histogram(
                salary_data,
                x='salary_avg',
                title="Salary Distribution",
                nbins=20
            )
            figures['salary_distribution'] = fig
            
            # Salary by experience
            if 'experience_category' in self.df.columns:
                salary_exp = self.df.dropna(subset=['salary_avg', 'experience_category'])
                if not salary_exp.empty:
                    fig = px.box(
                        salary_exp,
                        x='experience_category',
                        y='salary_avg',
                        title="Salary Distribution by Experience Level"
                    )
                    figures['salary_by_experience'] = fig
            
            # Salary by sector
            if 'sector' in self.df.columns:
                salary_sector = self.df.dropna(subset=['salary_avg', 'sector'])
                if not salary_sector.empty:
                    fig = px.box(
                        salary_sector,
                        x='sector',
                        y='salary_avg',
                        title="Salary Distribution by Sector"
                    )
                    fig.update_xaxis(tickangle=45)
                    figures['salary_by_sector'] = fig
        
        return figures
    
    def create_correlation_heatmap(self) -> Optional[go.Figure]:
        """Create correlation heatmap for numerical variables"""
        numerical_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numerical_cols) > 1:
            corr_matrix = self.df[numerical_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Correlation Matrix - Numerical Variables",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            return fig
        
        return None
    
    def create_wordcloud_figure(self) -> Optional[go.Figure]:
        """Create word cloud from job descriptions"""
        if 'description' not in self.df.columns:
            return None
        
        # Combine all descriptions
        text = ' '.join(self.df['description'].fillna('').astype(str))
        
        if len(text.strip()) == 0:
            return None
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        # Convert to plotly figure
        fig = go.Figure()
        fig.add_trace(go.Image(z=wordcloud.to_array()))
        fig.update_layout(
            title="Most Common Words in Job Descriptions",
            xaxis_showticklabels=False,
            yaxis_showticklabels=False,
            xaxis_showgrid=False,
            yaxis_showgrid=False
        )
        
        return fig
    
    def create_comprehensive_dashboard(self) -> Dict[str, go.Figure]:
        """Create a comprehensive set of visualizations"""
        logger.info("Creating comprehensive visualization dashboard")
        
        all_figures = {}
        
        # Distribution charts
        all_figures.update(self.create_distribution_charts())
        
        # Trend charts
        all_figures.update(self.create_trend_charts())
        
        # Skills charts
        all_figures.update(self.create_skills_charts())
        
        # Salary charts
        all_figures.update(self.create_salary_charts())
        
        # Correlation heatmap
        corr_fig = self.create_correlation_heatmap()
        if corr_fig:
            all_figures['correlation_heatmap'] = corr_fig
        
        # Word cloud
        wordcloud_fig = self.create_wordcloud_figure()
        if wordcloud_fig:
            all_figures['wordcloud'] = wordcloud_fig
        
        logger.info(f"Created {len(all_figures)} visualizations")
        return all_figures 