#!/usr/bin/env python3
"""
Complete Job Market Analysis Platform
All-in-one Streamlit application with scraping, EDA, topic modeling, and AI reports
"""
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import logging
from datetime import datetime
import json
import asyncio
import threading
import time
import requests
from typing import Dict, List, Optional

# Import our modules
from scrapers.maroc_annonce_scraper import MarocAnnonceScraper
from scrapers.rekrute_scraper import RekruteScraper
from config.database import DatabaseManager
from utils.topic_modeling import JobTopicModeler
from utils.report_generator import generate_comprehensive_report
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Job Market Analysis Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .section-card {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    .success-banner {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if 'scrapers_running' not in st.session_state:
    st.session_state.scrapers_running = False
if 'scraping_progress' not in st.session_state:
    st.session_state.scraping_progress = {}

def load_job_data():
    """Load and cache job market data"""
    try:
        conn = sqlite3.connect('data/job_market.db')
        query = """
        SELECT title, company, location, sector, fonction,
               experience, education_level, contract_type,
               salary, salary_min, salary_max,
               description, requirements, source,
               is_active, view_count, likes, date_scraped
        FROM jobs WHERE is_active = 1
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def run_scraper_async(scraper_name, scraper_class):
    """Run scraper asynchronously"""
    try:
        scraper = scraper_class()
        st.session_state.scraping_progress[scraper_name] = "🔄 Starting..."
        
        # Run scraper
        asyncio.run(scraper.scrape_all())
        
        st.session_state.scraping_progress[scraper_name] = "✅ Completed"
    except Exception as e:
        st.session_state.scraping_progress[scraper_name] = f"❌ Error: {str(e)}"
        logger.error(f"Scraper {scraper_name} failed: {e}")

def scraping_management_section():
    """Data scraping management interface"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 🔄 Data Collection & Scraping")
    
    st.markdown("""
    **Automated data collection from major Moroccan job platforms**
    
    Collect fresh job market data from MarocAnnonce and Rekrute to ensure up-to-date analysis.
    """)
    
    # Scraping controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎯 Scrape MarocAnnonce", disabled=st.session_state.scrapers_running):
            st.session_state.scrapers_running = True
            thread = threading.Thread(
                target=run_scraper_async, 
                args=("MarocAnnonce", MarocAnnonceScraper)
            )
            thread.start()
    
    with col2:
        if st.button("🔍 Scrape Rekrute", disabled=st.session_state.scrapers_running):
            st.session_state.scrapers_running = True
            thread = threading.Thread(
                target=run_scraper_async, 
                args=("Rekrute", RekruteScraper)
            )
            thread.start()
    
    with col3:
        if st.button("🚀 Scrape All Sources", disabled=st.session_state.scrapers_running):
            st.session_state.scrapers_running = True
            for scraper_name, scraper_class in [("MarocAnnonce", MarocAnnonceScraper), ("Rekrute", RekruteScraper)]:
                thread = threading.Thread(
                    target=run_scraper_async, 
                    args=(scraper_name, scraper_class)
                )
                thread.start()
    
    # Progress display
    if st.session_state.scraping_progress:
        st.markdown("### 📊 Scraping Progress")
        for scraper, status in st.session_state.scraping_progress.items():
            st.write(f"**{scraper}**: {status}")
        
        # Auto-refresh
        if st.session_state.scrapers_running:
            time.sleep(2)
            st.rerun()
        
        # Reset when all complete
        all_complete = all("✅" in status or "❌" in status for status in st.session_state.scraping_progress.values())
        if all_complete and st.session_state.scrapers_running:
            st.session_state.scrapers_running = False
            st.success("🎉 Scraping completed! Data updated.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def market_overview_section(df):
    """Market overview with key metrics"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Market Overview")
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_jobs = len(df)
        st.markdown(f'<div class="metric-card"><h3>📋 Total Jobs</h3><h2>{total_jobs:,}</h2></div>', unsafe_allow_html=True)
        
    with col2:
        unique_companies = df['company'].nunique()
        st.markdown(f'<div class="metric-card"><h3>🏢 Companies</h3><h2>{unique_companies:,}</h2></div>', unsafe_allow_html=True)
        
    with col3:
        unique_sectors = df['sector'].nunique()
        st.markdown(f'<div class="metric-card"><h3>🎯 Sectors</h3><h2>{unique_sectors:,}</h2></div>', unsafe_allow_html=True)
        
    with col4:
        unique_locations = df['location'].nunique()
        st.markdown(f'<div class="metric-card"><h3>📍 Locations</h3><h2>{unique_locations:,}</h2></div>', unsafe_allow_html=True)
    
    with col5:
        # Data freshness
        if 'date_scraped' in df.columns:
            latest_date = pd.to_datetime(df['date_scraped']).max()
            days_old = (datetime.now() - latest_date).days
            freshness = f"{days_old} days ago" if days_old > 0 else "Today"
        else:
            freshness = "Unknown"
        st.markdown(f'<div class="metric-card"><h3>📅 Data Freshness</h3><h2>{freshness}</h2></div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def comprehensive_eda_section(df):
    """Comprehensive EDA with all visualizations"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 📈 Comprehensive Market Analytics & EDA")
    
    # Create comprehensive dashboard
    tab1, tab2, tab3 = st.tabs(["📊 Market Insights", "🎯 Sector Analysis", "📋 Data Quality"])
    
    with tab1:
        # Market insights visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Source distribution
            source_counts = df['source'].value_counts()
            fig1 = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="📊 Job Distribution by Source",
                color_discrete_sequence=['#FF6B6B', '#4ECDC4']
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Location treemap
            location_counts = df['location'].value_counts().head(12)
            fig3 = px.treemap(
                names=location_counts.index,
                values=location_counts.values,
                title="🗺️ Geographic Job Distribution"
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Education requirements
            education_counts = df['education_level'].value_counts()
            fig5 = px.funnel(
                y=education_counts.index,
                x=education_counts.values,
                title="🎓 Education Level Requirements"
            )
            st.plotly_chart(fig5, use_container_width=True)
        
        with col2:
            # Top sectors
            sector_counts = df['sector'].value_counts().head(10)
            fig2 = px.bar(
                x=sector_counts.values,
                y=sector_counts.index,
                orientation='h',
                title="🏭 Top 10 Job Sectors",
                color=sector_counts.values,
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Contract types
            contract_counts = df['contract_type'].value_counts()
            fig4 = px.bar(
                x=contract_counts.index,
                y=contract_counts.values,
                title="📝 Contract Type Preferences",
                color=contract_counts.values,
                color_continuous_scale='plasma'
            )
            st.plotly_chart(fig4, use_container_width=True)
            
            # Company distribution
            company_counts = df['company'].value_counts().head(15)
            fig6 = px.scatter(
                x=range(len(company_counts)),
                y=company_counts.values,
                size=company_counts.values,
                hover_name=company_counts.index,
                title="🏢 Top Companies by Job Volume",
                labels={'x': 'Company Rank', 'y': 'Number of Jobs'}
            )
            st.plotly_chart(fig6, use_container_width=True)
    
    with tab2:
        # Sector deep dive
        st.markdown("### 🎯 Sector Deep Dive Analysis")
        
        # Sector vs location heatmap
        sector_location = pd.crosstab(df['sector'], df['location']).fillna(0)
        top_sectors = df['sector'].value_counts().head(10).index
        top_locations = df['location'].value_counts().head(8).index
        
        heatmap_data = sector_location.loc[top_sectors, top_locations]
        
        fig_heatmap = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            title="🌡️ Sector vs Location Heatmap",
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Salary analysis by sector
        if 'salary_min' in df.columns:
            salary_by_sector = df.groupby('sector')['salary_min'].mean().sort_values(ascending=False).head(10)
            fig_salary = px.bar(
                x=salary_by_sector.values,
                y=salary_by_sector.index,
                orientation='h',
                title="💰 Average Salary by Sector (Top 10)",
                color=salary_by_sector.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_salary, use_container_width=True)
    
    with tab3:
        # Data quality analysis
        st.markdown("### 🔍 Data Quality Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Completeness analysis
            completeness = {}
            for col in ['company', 'location', 'sector', 'description', 'requirements']:
                if col in df.columns:
                    completeness[col] = (1 - df[col].isna().sum() / len(df)) * 100
            
            fig_completeness = px.bar(
                x=list(completeness.keys()),
                y=list(completeness.values()),
                title="📊 Data Completeness by Field",
                color=list(completeness.values()),
                color_continuous_scale='RdYlGn'
            )
            fig_completeness.update_layout(yaxis_title="Completeness %")
            st.plotly_chart(fig_completeness, use_container_width=True)
        
        with col2:
            # Text quality analysis
            if 'description' in df.columns:
                desc_lengths = df['description'].str.len().dropna()
                req_lengths = df['requirements'].str.len().dropna()
                
                fig_text = go.Figure()
                fig_text.add_trace(go.Histogram(x=desc_lengths, name='Description Length', opacity=0.7))
                fig_text.add_trace(go.Histogram(x=req_lengths, name='Requirements Length', opacity=0.7))
                fig_text.update_layout(
                    title='📝 Text Content Quality Analysis',
                    xaxis_title='Character Length',
                    yaxis_title='Count',
                    barmode='overlay'
                )
                st.plotly_chart(fig_text, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def descriptive_statistics_section(df):
    """Generate and display descriptive statistics"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 📋 Descriptive Statistics & Data Summary")
    
    # Generate comprehensive stats
    stats = generate_descriptive_stats(df)
    
    # Display stats in organized tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Basic Stats", "💰 Salary Analysis", "🏢 Categorical Analysis", "📝 Text Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Dataset Overview")
            st.json(stats['basic_stats'])
        with col2:
            st.markdown("### ❌ Missing Values Analysis")
            st.json(stats['missing_values'])
    
    with tab2:
        if stats['salary_stats']:
            st.markdown("### 💰 Salary Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Minimum Salary Stats:**")
                st.json(stats['salary_stats']['min_salary'])
            with col2:
                st.markdown("**Maximum Salary Stats:**")
                st.json(stats['salary_stats']['max_salary'])
    
    with tab3:
        st.markdown("### 🏢 Categorical Data Distribution")
        for category, data in stats['categorical_stats'].items():
            with st.expander(f"📊 {category.title()} Distribution"):
                st.json(data)
    
    with tab4:
        st.markdown("### 📝 Text Content Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Description Statistics:**")
            st.json(stats['text_stats']['description'])
        with col2:
            st.markdown("**Requirements Statistics:**")
            st.json(stats['text_stats']['requirements'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    return stats

def generate_descriptive_stats(df):
    """Generate comprehensive descriptive statistics"""
    stats = {
        'basic_stats': {
            'total_jobs': len(df),
            'unique_companies': df['company'].nunique(),
            'unique_locations': df['location'].nunique(),
            'unique_sectors': df['sector'].nunique(),
        },
        'missing_values': {
            field: int(df[field].isna().sum()) for field in ['company', 'location', 'sector', 'description', 'requirements']
            if field in df.columns
        },
        'salary_stats': {},
        'categorical_stats': {},
        'text_stats': {}
    }
    
    # Salary stats
    if 'salary_min' in df.columns:
        salary_min = df['salary_min'].dropna()
        salary_max = df['salary_max'].dropna()
        
        stats['salary_stats'] = {
            'min_salary': {
                'mean': float(salary_min.mean()) if not salary_min.empty else None,
                'median': float(salary_min.median()) if not salary_min.empty else None,
                'std': float(salary_min.std()) if not salary_min.empty else None,
                'min': float(salary_min.min()) if not salary_min.empty else None,
                'max': float(salary_min.max()) if not salary_min.empty else None
            },
            'max_salary': {
                'mean': float(salary_max.mean()) if not salary_max.empty else None,
                'median': float(salary_max.median()) if not salary_max.empty else None,
                'std': float(salary_max.std()) if not salary_max.empty else None,
                'min': float(salary_max.min()) if not salary_max.empty else None,
                'max': float(salary_max.max()) if not salary_max.empty else None
            }
        }
    
    # Categorical stats
    categorical_fields = ['source', 'sector', 'location', 'contract_type', 'education_level', 'experience']
    for field in categorical_fields:
        if field in df.columns:
            stats['categorical_stats'][field] = df[field].value_counts().head(10).to_dict()
    
    # Text stats
    text_fields = ['description', 'requirements']
    for field in text_fields:
        if field in df.columns:
            text_data = df[field].dropna()
            stats['text_stats'][field] = {
                'avg_length': float(text_data.str.len().mean()) if not text_data.empty else None,
                'min_length': int(text_data.str.len().min()) if not text_data.empty else None,
                'max_length': int(text_data.str.len().max()) if not text_data.empty else None,
                'total_words': int(text_data.str.split().str.len().sum()) if not text_data.empty else None,
                'avg_words': float(text_data.str.split().str.len().mean()) if not text_data.empty else None
            }
    
    return stats

def topic_modeling_section(df):
    """Advanced topic modeling interface"""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 🤖 Advanced Topic Modeling & AI Analysis")
    
    st.markdown("""
    **Discover hidden patterns in job descriptions using machine learning!**
    
    This advanced analysis uses unsupervised learning to identify distinct job market segments,
    complete with AI-powered insights and downloadable reports.
    """)
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_options = ["All Sources", "MarocAnnonce", "Rekrute"]
        selected_source = st.selectbox("🎯 Data Source", source_options)
        
    with col2:
        algorithm_options = ["KMeans", "DBSCAN"]
        selected_algorithm = st.selectbox("⚙️ Algorithm", algorithm_options)
        
    with col3:
        if selected_algorithm == "KMeans":
            n_clusters = st.slider("📊 Clusters", 2, 10, 5)
        else:
            eps = st.slider("🎯 Epsilon", 0.1, 2.0, 0.5, 0.1)
            min_samples = st.slider("👥 Min Samples", 2, 20, 5)
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        text_source = st.selectbox("Text Source", ["combined", "description", "requirements"])
        max_features = st.slider("Max Features", 100, 2000, 1000)
    
    # Run analysis
    if st.button("🚀 Run Topic Modeling Analysis", type="primary"):
        with st.spinner("🔄 Performing advanced analysis..."):
            try:
                # Initialize and run topic modeler
                modeler = JobTopicModeler()
                
                if modeler.load_data(source_filter=selected_source if selected_source != "All Sources" else None):
                    st.success(f"✅ Loaded {len(modeler.df)} records")
                    
                    if modeler.preprocess_text(text_column=text_source):
                        if modeler.create_features(max_features=max_features):
                            # Run clustering
                            if selected_algorithm == "KMeans":
                                success = modeler.fit_kmeans(n_clusters=n_clusters)
                            else:
                                success = modeler.fit_dbscan(eps=eps, min_samples=min_samples)
                            
                            if success:
                                st.success(f"✅ {selected_algorithm} completed!")
                                display_topic_results(modeler)
                                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_topic_results(modeler):
    """Display topic modeling results"""
    st.markdown("### 📊 Topic Modeling Results")
    
    # Basic metrics
    n_clusters = len(set(modeler.clusters)) - (1 if -1 in modeler.clusters else 0)
    n_noise = list(modeler.clusters).count(-1) if -1 in modeler.clusters else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Clusters", n_clusters)
    with col2:
        st.metric("📄 Documents", len(modeler.df))
    with col3:
        st.metric("🔍 Noise Points", n_noise)
    
    # Visualizations
    tab1, tab2 = st.tabs(["🎨 Visualization", "📋 Cluster Details"])
    
    with tab1:
        cluster_viz = modeler.visualize_clusters()
        if cluster_viz:
            st.plotly_chart(cluster_viz, use_container_width=True)
    
    with tab2:
        cluster_topics = modeler.extract_cluster_topics()
        cluster_chars = modeler.analyze_cluster_characteristics()
        
        for cluster_id in sorted(cluster_topics.keys()):
            with st.expander(f"📁 Cluster {cluster_id} ({cluster_topics[cluster_id]['size']} jobs)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔑 Keywords:**")
                    st.write(", ".join(cluster_topics[cluster_id]['keywords'][:10]))
                    
                    st.markdown("**🏢 Top Sectors:**")
                    for sector, count in list(cluster_chars[cluster_id]['top_sectors'].items())[:3]:
                        st.write(f"• {sector}: {count}")
                
                with col2:
                    st.markdown("**📍 Locations:**")
                    for location, count in list(cluster_chars[cluster_id]['top_locations'].items())[:3]:
                        st.write(f"• {location}: {count}")
                    
                    st.markdown("**📊 Sources:**")
                    for source, count in cluster_chars[cluster_id]['source_distribution'].items():
                        st.write(f"• {source}: {count}")
    
    # Report generation
    st.markdown("### 📑 AI-Powered Report Generation")
    if st.button("📄 Generate Comprehensive Report", type="primary"):
        generate_ai_report(modeler)

def generate_ai_report(modeler):
    """Generate AI-powered comprehensive report"""
    with st.spinner("🤖 Generating AI analysis and report..."):
        try:
            # Get report data
            report_data = modeler.generate_report_data()
            
            # Add Llama4 summary via Grok
            grok_summary = generate_grok_llama4_summary(report_data)
            if grok_summary:
                report_data['ai_summary'] = grok_summary
            
            # Generate PDF
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f'reports/comprehensive_analysis_{timestamp}.pdf'
            
            if generate_comprehensive_report(report_data, output_path):
                st.success("✅ Report generated successfully!")
                
                # Download buttons
                if os.path.exists(output_path):
                    with open(output_path, "rb") as file:
                        st.download_button(
                            "⬇️ Download PDF Report",
                            file,
                            file_name=f"job_market_analysis_{timestamp}.pdf",
                            mime="application/pdf"
                        )
                
                # Display summary
                if grok_summary:
                    st.markdown("### 🤖 AI-Generated Summary (Llama4)")
                    st.markdown(grok_summary)
                    
        except Exception as e:
            st.error(f"❌ Report generation failed: {str(e)}")

def generate_grok_llama4_summary(report_data):
    """Generate summary using Llama4 via Grok API"""
    grok_api_key = os.getenv('GROQ_API_KEY')
    if not grok_api_key:
        return None
    
    try:
        prompt = f"""
Using the Llama4 model, provide a concise executive summary of this job market analysis:

Analysis Results:
- {report_data['metadata']['n_clusters']} clusters identified from {report_data['metadata']['total_documents']} job postings
- Algorithm used: {report_data['metadata']['algorithm']}

Key Findings:
{json.dumps(report_data['summary_insights'], indent=2)}

Please provide a 3-paragraph executive summary covering:
1. Market overview and key patterns
2. Strategic implications for stakeholders
3. Future recommendations

Keep it professional and actionable for business decision-makers.
"""

        headers = {
            'Authorization': f'Bearer {grok_api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
            'messages': [
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.7,
            'max_completion_tokens': 500
        }
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            choices = response.json().get('choices', [])
            if choices:
                return choices[0]['message']['content']
            return response.json().get('text', '')
        else:
            logger.error(f"Grok API error {response.status_code}: {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error calling Grok API: {e}")
        return None

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Job Market Analysis Platform</h1>
        <p>Complete end-to-end analysis of Moroccan job market data with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_job_data()
    
    # Navigation
    page = st.sidebar.selectbox(
        "📋 Navigation",
        ["🏠 Overview", "🔄 Data Collection", "📊 Market Analytics", "📋 Descriptive Stats", "🤖 Topic Modeling"]
    )
    
    if page == "🏠 Overview":
        if not df.empty:
            market_overview_section(df)
            
            # Quick insights
            st.markdown('<div class="success-banner">', unsafe_allow_html=True)
            st.markdown(f"""
            **🎉 Platform Ready!** 
            - **{len(df):,} jobs** analyzed across **{df['sector'].nunique()} sectors**
            - Data from **{', '.join(df['source'].unique())}**
            - **{df['location'].nunique()} locations** covered
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add Download Report button
            st.markdown("### 📑 Comprehensive Report")
            if st.button("📄 Download Detailed Report", type="primary"):
                with st.spinner("🤖 Generating comprehensive report..."):
                    try:
                        # Generate report data
                        modeler = JobTopicModeler()
                        if modeler.load_data():
                            modeler.preprocess_text('combined')
                            modeler.create_features()
                            modeler.fit_kmeans(n_clusters=5)  # Example with KMeans
                            report_data = modeler.generate_report_data()
                            
                            # Add Llama4 summary via Grok
                            grok_summary = generate_grok_llama4_summary(report_data)
                            if grok_summary:
                                report_data['ai_summary'] = grok_summary
                            
                            # Generate PDF
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            output_path = f'reports/comprehensive_analysis_{timestamp}.pdf'
                            
                            if generate_comprehensive_report(report_data, output_path):
                                st.success("✅ Report generated successfully!")
                                
                                # Download button
                                if os.path.exists(output_path):
                                    with open(output_path, "rb") as file:
                                        st.download_button(
                                            "⬇️ Download PDF Report",
                                            file,
                                            file_name=f"job_market_analysis_{timestamp}.pdf",
                                            mime="application/pdf"
                                        )
                    except Exception as e:
                        st.error(f"❌ Report generation failed: {str(e)}")
        else:
            st.warning("⚠️ No data available. Please start with Data Collection.")
    
    elif page == "�� Data Collection":
        scraping_management_section()
    
    elif page == "📊 Market Analytics":
        if not df.empty:
            comprehensive_eda_section(df)
        else:
            st.warning("⚠️ No data available for analysis.")
    
    elif page == "📋 Descriptive Stats":
        if not df.empty:
            descriptive_statistics_section(df)
        else:
            st.warning("⚠️ No data available for statistics.")
    
    elif page == "🤖 Topic Modeling":
        if not df.empty:
            topic_modeling_section(df)
        else:
            st.warning("⚠️ No data available for topic modeling.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚀 Job Market Analysis Platform | Built with Streamlit, ML & AI</p>
        <p>Data sources: MarocAnnonce & Rekrute | AI powered by Grok & Llama4</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 