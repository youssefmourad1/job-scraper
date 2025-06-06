"""
Professional Job Market Analyzer - Main Streamlit Application
"""
import streamlit as st
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Initialize logging first
from utils.logger import setup_logging
setup_logging()

# Import modules
from config.settings import DASHBOARD_CONFIG, ANALYTICS_CONFIG
from config.database import DatabaseManager
from scrapers import RekruteScraper, MarocAnnonceScraper
from analytics.data_processor import DataProcessor
from utils.helpers import format_number, get_date_range_filter

# Configure Streamlit page
st.set_page_config(
    page_title=DASHBOARD_CONFIG['page_title'],
    page_icon=DASHBOARD_CONFIG['page_icon'],
    layout=DASHBOARD_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.18);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    
    .scraper-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border: 1px solid #e9ecef;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
logger = logging.getLogger(__name__)
db_manager = DatabaseManager()
data_processor = DataProcessor()

def init_session_state():
    """Initialize session state variables"""
    if 'scrapers_initialized' not in st.session_state:
        st.session_state.scrapers_initialized = False
    if 'scraping_in_progress' not in st.session_state:
        st.session_state.scraping_in_progress = False
    if 'last_scraping_results' not in st.session_state:
        st.session_state.last_scraping_results = {}

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create a styled metric card"""
    delta_html = ""
    if delta:
        color = {"normal": "#666", "positive": "#28a745", "negative": "#dc3545"}[delta_color]
        delta_html = f'<p style="color: {color}; font-size: 0.9rem; margin: 0;">{delta}</p>'
    
    return f"""
    <div class="metric-card">
        <h3 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.1rem;">{title}</h3>
        <h2 style="color: #667eea; margin: 0; font-size: 2rem; font-weight: bold;">{value}</h2>
        {delta_html}
    </div>
    """

def display_dashboard_header():
    """Display the main dashboard header"""
    st.markdown('<h1 class="main-header">🚀 Professional Job Market Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #6c757d;">
            Advanced job market intelligence platform for Morocco
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_key_metrics():
    """Display key performance metrics"""
    st.markdown('<h2 class="section-header">📊 Market Overview</h2>', unsafe_allow_html=True)
    
    # Get market statistics
    stats = db_manager.get_market_stats()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            create_metric_card(
                "Total Jobs", 
                format_number(stats['total_jobs']),
                f"+{stats['jobs_last_week']} this week" if stats['jobs_last_week'] > 0 else None,
                "positive" if stats['jobs_last_week'] > 0 else "normal"
            ), 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            create_metric_card("Companies", format_number(stats['unique_companies'])), 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            create_metric_card("Sectors", format_number(stats['unique_sectors'])), 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            create_metric_card("Locations", format_number(stats['unique_locations'])), 
            unsafe_allow_html=True
        )

def display_scraping_interface():
    """Display the scraping control interface"""
    st.markdown('<h2 class="section-header">🔄 Data Collection</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="scraper-card">
            <h4>🎯 Rekrute.com</h4>
            <p>Professional job platform with detailed job descriptions and company information.</p>
        </div>
        """, unsafe_allow_html=True)
        
        rekrute_enabled = st.checkbox("Enable Rekrute Scraper", value=True, key="rekrute_enabled")
    
    with col2:
        st.markdown("""
        <div class="scraper-card">
            <h4>🏢 MarocAnnonces.com</h4>
            <p>General classified platform with diverse job opportunities across Morocco.</p>
        </div>
        """, unsafe_allow_html=True)
        
        maroc_enabled = st.checkbox("Enable MarocAnnonces Scraper", value=True, key="maroc_enabled")
    
    # Scraping configuration
    st.markdown("### ⚙️ Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_pages = st.slider("Max Pages per Scraper", 1, 50, 10, key="max_pages")
    
    with col2:
        parallel_workers = st.slider("Parallel Workers", 1, 10, 4, key="parallel_workers")
    
    with col3:
        request_delay = st.slider("Request Delay (seconds)", 0.5, 5.0, 1.0, key="request_delay")
    
    # Scraping controls
    st.markdown("### 🚀 Launch Scraping")
    
    if not st.session_state.scraping_in_progress:
        if st.button("🔥 Start Data Collection", type="primary", use_container_width=True):
            run_scraping_session(rekrute_enabled, maroc_enabled, max_pages)
    else:
        st.markdown("""
        <div class="status-info">
            <strong>🔄 Scraping in Progress...</strong><br>
            Please wait while we collect the latest job data.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⏹️ Stop Scraping", type="secondary"):
            st.session_state.scraping_in_progress = False
            st.experimental_rerun()

def run_scraping_session(rekrute_enabled: bool, maroc_enabled: bool, max_pages: int):
    """Run the scraping session"""
    st.session_state.scraping_in_progress = True
    
    progress_container = st.container()
    status_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    results = {}
    total_scrapers = sum([rekrute_enabled, maroc_enabled])
    current_scraper = 0
    
    # Run Rekrute scraper
    if rekrute_enabled:
        with status_container:
            st.info("🎯 Scraping Rekrute.com...")
        
        try:
            rekrute_scraper = RekruteScraper()
            
            def update_progress(completed, total, jobs_found):
                progress = (current_scraper + completed/total) / total_scrapers
                progress_bar.progress(progress)
                status_text.text(f"Rekrute: {completed}/{total} pages, {jobs_found} jobs found")
            
            session = rekrute_scraper.run_sync(max_pages, update_progress)
            results['Rekrute'] = {
                'status': session.status,
                'new_jobs': session.new_jobs,
                'total_jobs': session.total_jobs,
                'errors': session.errors
            }
            
        except Exception as e:
            results['Rekrute'] = {'status': 'failed', 'error': str(e)}
        
        current_scraper += 1
    
    # Run MarocAnnonces scraper
    if maroc_enabled:
        with status_container:
            st.info("🏢 Scraping MarocAnnonces.com...")
        
        try:
            maroc_scraper = MarocAnnonceScraper()
            
            def update_progress(completed, total, jobs_found):
                progress = (current_scraper + completed/total) / total_scrapers
                progress_bar.progress(progress)
                status_text.text(f"MarocAnnonces: {completed}/{total} pages, {jobs_found} jobs found")
            
            session = maroc_scraper.run_sync(max_pages, update_progress)
            results['MarocAnnonces'] = {
                'status': session.status,
                'new_jobs': session.new_jobs,
                'total_jobs': session.total_jobs,
                'errors': session.errors
            }
            
        except Exception as e:
            results['MarocAnnonces'] = {'status': 'failed', 'error': str(e)}
        
        current_scraper += 1
    
    # Complete progress
    progress_bar.progress(1.0)
    status_text.text("✅ Scraping completed!")
    
    # Display results
    display_scraping_results(results)
    
    st.session_state.scraping_in_progress = False
    st.session_state.last_scraping_results = results

def display_scraping_results(results: Dict):
    """Display scraping session results"""
    st.markdown("### 📋 Scraping Results")
    
    for source, result in results.items():
        if result['status'] == 'completed':
            st.markdown(f"""
            <div class="status-success">
                <strong>✅ {source}</strong><br>
                New jobs: {result['new_jobs']} | Total processed: {result['total_jobs']} | Errors: {result['errors']}
            </div>
            """, unsafe_allow_html=True)
        else:
            error_msg = result.get('error', 'Unknown error')
            st.markdown(f"""
            <div class="status-warning">
                <strong>⚠️ {source}</strong><br>
                Status: {result['status']} | Error: {error_msg}
            </div>
            """, unsafe_allow_html=True)

def display_analytics_dashboard():
    """Display the analytics dashboard"""
    st.markdown('<h2 class="section-header">📈 Market Analytics</h2>', unsafe_allow_html=True)
    
    # Load and process data
    with st.spinner("Loading and processing data..."):
        df = data_processor.load_and_clean_data()
    
    if df.empty:
        st.warning("No data available. Please run the scrapers first.")
        return
    
    # Date filter
    st.markdown("### 📅 Filter Data")
    col1, col2 = st.columns(2)
    
    with col1:
        days_filter = st.selectbox(
            "Time Period", 
            [7, 30, 60, 90, 365], 
            index=1,
            format_func=lambda x: f"Last {x} days"
        )
    
    with col2:
        source_filter = st.multiselect(
            "Data Sources",
            options=df['source'].unique(),
            default=df['source'].unique()
        )
    
    # Apply filters
    date_filter = get_date_range_filter(days_filter)
    if source_filter:
        df = df[df['source'].isin(source_filter)]
    
    # Display visualizations
    display_job_distribution_charts(df)
    display_market_trends_charts(df)
    display_skills_analysis(df)
    display_salary_analysis(df)

def display_job_distribution_charts(df: pd.DataFrame):
    """Display job distribution visualizations"""
    st.markdown("### 🗺️ Job Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Source distribution
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            fig_source = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Jobs by Source",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_source.update_layout(height=400)
            st.plotly_chart(fig_source, use_container_width=True)
    
    with col2:
        # Location distribution
        if 'location' in df.columns:
            location_counts = df['location'].value_counts().head(10)
            fig_location = px.bar(
                x=location_counts.values,
                y=location_counts.index,
                orientation='h',
                title="Top 10 Locations",
                color=location_counts.values,
                color_continuous_scale="viridis"
            )
            fig_location.update_layout(height=400)
            st.plotly_chart(fig_location, use_container_width=True)
    
    # Sector distribution
    if 'sector' in df.columns:
        sector_counts = df['sector'].value_counts().head(15)
        fig_sector = px.treemap(
            names=sector_counts.index,
            values=sector_counts.values,
            title="Job Distribution by Sector"
        )
        fig_sector.update_layout(height=500)
        st.plotly_chart(fig_sector, use_container_width=True)

def display_market_trends_charts(df: pd.DataFrame):
    """Display market trends visualizations"""
    st.markdown("### 📊 Market Trends")
    
    if 'date_posted' in df.columns:
        # Daily job postings
        daily_posts = df.groupby(df['date_posted'].dt.date).size()
        fig_daily = px.line(
            x=daily_posts.index,
            y=daily_posts.values,
            title="Daily Job Postings Trend",
            labels={'x': 'Date', 'y': 'Number of Posts'}
        )
        fig_daily.update_layout(height=400)
        st.plotly_chart(fig_daily, use_container_width=True)
        
        # Weekly trends by sector
        if 'sector' in df.columns:
            df['week'] = df['date_posted'].dt.to_period('W')
            weekly_sector = df.groupby(['week', 'sector']).size().reset_index(name='count')
            weekly_sector['week'] = weekly_sector['week'].astype(str)
            
            top_sectors = df['sector'].value_counts().head(5).index
            weekly_sector_filtered = weekly_sector[weekly_sector['sector'].isin(top_sectors)]
            
            fig_weekly = px.line(
                weekly_sector_filtered,
                x='week',
                y='count',
                color='sector',
                title="Weekly Job Postings by Top Sectors",
                labels={'week': 'Week', 'count': 'Number of Posts'}
            )
            fig_weekly.update_layout(height=400)
            st.plotly_chart(fig_weekly, use_container_width=True)

def display_skills_analysis(df: pd.DataFrame):
    """Display skills analysis"""
    st.markdown("### 🎯 Skills Analysis")
    
    if 'extracted_skills' in df.columns:
        # Most demanded skills
        all_skills = []
        for skills_list in df['extracted_skills'].dropna():
            if isinstance(skills_list, list):
                all_skills.extend(skills_list)
        
        if all_skills:
            skill_counts = pd.Series(all_skills).value_counts().head(20)
            
            fig_skills = px.bar(
                x=skill_counts.values,
                y=skill_counts.index,
                orientation='h',
                title="Top 20 Most Demanded Skills",
                color=skill_counts.values,
                color_continuous_scale="plasma"
            )
            fig_skills.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_skills, use_container_width=True)

def display_salary_analysis(df: pd.DataFrame):
    """Display salary analysis"""
    st.markdown("### 💰 Salary Analysis")
    
    if 'salary_avg' in df.columns and df['salary_avg'].notna().sum() > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary distribution
            fig_salary_dist = px.histogram(
                df.dropna(subset=['salary_avg']),
                x='salary_avg',
                title="Salary Distribution",
                nbins=20
            )
            fig_salary_dist.update_layout(height=400)
            st.plotly_chart(fig_salary_dist, use_container_width=True)
        
        with col2:
            # Salary by experience
            if 'experience_category' in df.columns:
                salary_exp = df.dropna(subset=['salary_avg', 'experience_category'])
                if not salary_exp.empty:
                    fig_salary_exp = px.box(
                        salary_exp,
                        x='experience_category',
                        y='salary_avg',
                        title="Salary by Experience Level"
                    )
                    fig_salary_exp.update_layout(height=400)
                    st.plotly_chart(fig_salary_exp, use_container_width=True)

def main():
    """Main application function"""
    init_session_state()
    
    # Display header
    display_dashboard_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")
        page = st.radio(
            "Select Page",
            ["📊 Dashboard", "🔄 Data Collection", "📈 Analytics", "📋 Data Management"],
            key="navigation"
        )
        
        st.markdown("---")
        st.markdown("## 📊 Quick Stats")
        stats = db_manager.get_market_stats()
        st.metric("Total Jobs", format_number(stats['total_jobs']))
        st.metric("Companies", format_number(stats['unique_companies']))
        st.metric("This Week", format_number(stats['jobs_last_week']))
    
    # Main content based on navigation
    if page == "📊 Dashboard":
        display_key_metrics()
        st.markdown("---")
        display_analytics_dashboard()
    
    elif page == "🔄 Data Collection":
        display_scraping_interface()
    
    elif page == "📈 Analytics":
        display_analytics_dashboard()
    
    elif page == "📋 Data Management":
        display_data_management()

def display_data_management():
    """Display data management interface"""
    st.markdown('<h2 class="section-header">📋 Data Management</h2>', unsafe_allow_html=True)
    
    # Load data
    df = data_processor.load_and_clean_data()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Data overview
    st.markdown("### 📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Data Sources", df['source'].nunique())
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    
    # Data sample
    st.markdown("### 🔍 Data Sample")
    st.dataframe(df.head(100), use_container_width=True, height=400)
    
    # Export options
    st.markdown("### 📥 Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"job_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Jobs', index=False)
        
        st.download_button(
            label="📋 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"job_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.ms-excel"
        )
    
    with col3:
        # JSON export
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"job_data_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

                color_continuous_scale="plasma"
            )
            fig_skills.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_skills, use_container_width=True)

def display_salary_analysis(df: pd.DataFrame):
    """Display salary analysis"""
    st.markdown("### 💰 Salary Analysis")
    
    if 'salary_avg' in df.columns and df['salary_avg'].notna().sum() > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Salary distribution
            fig_salary_dist = px.histogram(
                df.dropna(subset=['salary_avg']),
                x='salary_avg',
                title="Salary Distribution",
                nbins=20
            )
            fig_salary_dist.update_layout(height=400)
            st.plotly_chart(fig_salary_dist, use_container_width=True)
        
        with col2:
            # Salary by experience
            if 'experience_category' in df.columns:
                salary_exp = df.dropna(subset=['salary_avg', 'experience_category'])
                if not salary_exp.empty:
                    fig_salary_exp = px.box(
                        salary_exp,
                        x='experience_category',
                        y='salary_avg',
                        title="Salary by Experience Level"
                    )
                    fig_salary_exp.update_layout(height=400)
                    st.plotly_chart(fig_salary_exp, use_container_width=True)

def main():
    """Main application function"""
    init_session_state()
    
    # Display header
    display_dashboard_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")
        page = st.radio(
            "Select Page",
            ["📊 Dashboard", "🔄 Data Collection", "📈 Analytics", "📋 Data Management"],
            key="navigation"
        )
        
        st.markdown("---")
        st.markdown("## 📊 Quick Stats")
        stats = db_manager.get_market_stats()
        st.metric("Total Jobs", format_number(stats['total_jobs']))
        st.metric("Companies", format_number(stats['unique_companies']))
        st.metric("This Week", format_number(stats['jobs_last_week']))
    
    # Main content based on navigation
    if page == "📊 Dashboard":
        display_key_metrics()
        st.markdown("---")
        display_analytics_dashboard()
    
    elif page == "🔄 Data Collection":
        display_scraping_interface()
    
    elif page == "📈 Analytics":
        display_analytics_dashboard()
    
    elif page == "📋 Data Management":
        display_data_management()

def display_data_management():
    """Display data management interface"""
    st.markdown('<h2 class="section-header">📋 Data Management</h2>', unsafe_allow_html=True)
    
    # Load data
    df = data_processor.load_and_clean_data()
    
    if df.empty:
        st.warning("No data available.")
        return
    
    # Data overview
    st.markdown("### 📊 Data Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Data Sources", df['source'].nunique())
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
    
    # Data sample
    st.markdown("### 🔍 Data Sample")
    st.dataframe(df.head(100), use_container_width=True, height=400)
    
    # Export options
    st.markdown("### 📥 Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"job_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Excel export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Jobs', index=False)
        
        st.download_button(
            label="📋 Download Excel",
            data=excel_buffer.getvalue(),
            file_name=f"job_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.ms-excel"
        )
    
    with col3:
        # JSON export
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📄 Download JSON",
            data=json_data,
            file_name=f"job_data_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main() 