"""
Professional Job Market Analyzer - Main Application
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import asyncio
from typing import Dict

# Setup
from utils.logger import setup_logging
setup_logging()

try:
    from config.database import DatabaseManager
    from scrapers import RekruteScraper, MarocAnnonceScraper
    from analytics.data_processor import DataProcessor
    from utils.helpers import format_number
    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Some modules could not be loaded. Please check your installation.")
    MODULES_LOADED = False
    
    # Fallback imports
    DatabaseManager = None
    DataProcessor = None
    format_number = lambda x: f"{x:,}" if x else "0"

# Page config
st.set_page_config(
    page_title="🚀 Job Market Analyzer",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
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
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
if MODULES_LOADED:
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = DataProcessor()
if 'scraping_in_progress' not in st.session_state:
    st.session_state.scraping_in_progress = False

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🚀 Professional Job Market Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #6c757d;">
            Advanced job market intelligence platform for Morocco
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if not MODULES_LOADED:
        st.error("⚠️ Some required modules could not be loaded. Please ensure all dependencies are installed.")
        st.info("Run: `pip install -r requirements.txt` to install missing dependencies.")
        return
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")
        page = st.radio(
            "Select Page",
            ["📊 Dashboard", "🔄 Data Collection", "📈 Analytics", "📋 Data Management"]
        )
        
        # Quick stats
        st.markdown("---")
        st.markdown("## 📊 Quick Stats")
        try:
            if hasattr(st.session_state, 'db_manager') and st.session_state.db_manager:
                stats = st.session_state.db_manager.get_market_stats()
                st.metric("Total Jobs", format_number(stats['total_jobs']))
                st.metric("Companies", format_number(stats['unique_companies']))
                st.metric("This Week", format_number(stats['jobs_last_week']))
            else:
                st.info("No data available yet")
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")
    
    # Main content
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🔄 Data Collection":
        show_data_collection()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "📋 Data Management":
        show_data_management()

def show_dashboard():
    """Display the main dashboard"""
    st.markdown('<h2 class="section-header">📊 Market Overview</h2>', unsafe_allow_html=True)
    
    if not MODULES_LOADED:
        st.warning("Dashboard requires all modules to be loaded.")
        return
    
    try:
        if hasattr(st.session_state, 'db_manager') and st.session_state.db_manager:
            stats = st.session_state.db_manager.get_market_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Jobs", format_number(stats['total_jobs']))
            with col2:
                st.metric("Companies", format_number(stats['unique_companies']))
            with col3:
                st.metric("Sectors", format_number(stats['unique_sectors']))
            with col4:
                st.metric("Locations", format_number(stats['unique_locations']))
            
            # Load and display basic analytics
            if hasattr(st.session_state, 'data_processor') and st.session_state.data_processor:
                df = st.session_state.data_processor.load_and_clean_data()
                if not df.empty:
                    st.markdown("### 📈 Recent Activity")
                    
                    # Basic charts
                    import plotly.express as px
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'source' in df.columns:
                            source_counts = df['source'].value_counts()
                            fig = px.pie(values=source_counts.values, names=source_counts.index, 
                                       title="Jobs by Source")
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'location' in df.columns:
                            location_counts = df['location'].value_counts().head(10)
                            fig = px.bar(x=location_counts.values, y=location_counts.index,
                                       orientation='h', title="Top Locations")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data available. Please run the scrapers first.")
            else:
                st.warning("Data processor not available.")
        else:
            st.info("Database not initialized yet.")
        
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

def show_data_collection():
    """Display data collection interface"""
    st.markdown('<h2 class="section-header">🔄 Data Collection</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Rekrute.com")
        st.markdown("Professional job platform with detailed descriptions")
        rekrute_enabled = st.checkbox("Enable Rekrute Scraper", value=True)
    
    with col2:
        st.markdown("### 🏢 MarocAnnonces.com")
        st.markdown("General classified platform with diverse opportunities")
        maroc_enabled = st.checkbox("Enable MarocAnnonces Scraper", value=True)
    
    # Configuration
    st.markdown("### ⚙️ Configuration")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_pages = st.slider("Max Pages per Scraper", 1, 50, 5)
    with col2:
        max_workers = st.slider("Parallel Workers", 1, 8, 4)
    with col3:
        request_delay = st.slider("Request Delay (seconds)", 0.5, 5.0, 1.0)
    
    # Scraping controls
    st.markdown("### 🚀 Launch Scraping")
    
    if not st.session_state.scraping_in_progress:
        if st.button("🔥 Start Data Collection", type="primary", use_container_width=True):
            run_scraping(rekrute_enabled, maroc_enabled, max_pages)
    else:
        st.info("🔄 Scraping in progress...")
        if st.button("⏹️ Stop Scraping"):
            st.session_state.scraping_in_progress = False
            st.rerun()

def run_scraping(rekrute_enabled: bool, maroc_enabled: bool, max_pages: int):
    """Run the scraping process"""
    st.session_state.scraping_in_progress = True
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results = {}
    
    try:
        scrapers_to_run = []
        if rekrute_enabled:
            scrapers_to_run.append(('Rekrute', RekruteScraper()))
        if maroc_enabled:
            scrapers_to_run.append(('MarocAnnonces', MarocAnnonceScraper()))
        
        for i, (name, scraper) in enumerate(scrapers_to_run):
            status_text.text(f"Scraping {name}...")
            
            try:
                # Simple synchronous scraping for demo
                # In production, you would use the async methods
                session = scraper.run_sync(max_pages)
                results[name] = {
                    'status': session.status,
                    'new_jobs': session.new_jobs,
                    'total_jobs': session.total_jobs,
                    'errors': session.errors
                }
            except Exception as e:
                results[name] = {'status': 'failed', 'error': str(e)}
            
            progress_bar.progress((i + 1) / len(scrapers_to_run))
        
        # Display results
        status_text.text("✅ Scraping completed!")
        
        for source, result in results.items():
            if result['status'] == 'completed':
                st.success(f"✅ {source}: {result['new_jobs']} new jobs, {result['total_jobs']} total")
            else:
                st.error(f"❌ {source}: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        st.error(f"Scraping failed: {str(e)}")
    
    finally:
        st.session_state.scraping_in_progress = False

def show_analytics():
    """Display analytics dashboard"""
    st.markdown('<h2 class="section-header">📈 Market Analytics</h2>', unsafe_allow_html=True)
    
    try:
        df = st.session_state.data_processor.load_and_clean_data()
        
        if df.empty:
            st.warning("No data available. Please run the scrapers first.")
            return
        
        # Data overview
        st.markdown("### 📊 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Date Range", f"{(df['date_posted'].max() - df['date_posted'].min()).days} days" 
                     if 'date_posted' in df.columns else "N/A")
        with col3:
            st.metric("Sources", df['source'].nunique() if 'source' in df.columns else 0)
        with col4:
            st.metric("Sectors", df['sector'].nunique() if 'sector' in df.columns else 0)
        
        # Advanced analytics
        if st.checkbox("Show Advanced Analytics"):
            import plotly.express as px
            
            # Skills analysis
            if 'extracted_skills' in df.columns:
                st.markdown("### 🎯 Skills Analysis")
                all_skills = []
                for skills in df['extracted_skills'].dropna():
                    if isinstance(skills, list):
                        all_skills.extend(skills)
                
                if all_skills:
                    skill_counts = pd.Series(all_skills).value_counts().head(15)
                    fig = px.bar(x=skill_counts.values, y=skill_counts.index, 
                               orientation='h', title="Most Demanded Skills")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Salary analysis
            if 'salary_avg' in df.columns:
                st.markdown("### 💰 Salary Analysis")
                salary_data = df.dropna(subset=['salary_avg'])
                if not salary_data.empty:
                    fig = px.histogram(salary_data, x='salary_avg', title="Salary Distribution")
                    st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")

def show_data_management():
    """Display data management interface"""
    st.markdown('<h2 class="section-header">📋 Data Management</h2>', unsafe_allow_html=True)
    
    try:
        df = st.session_state.data_processor.load_and_clean_data()
        
        if df.empty:
            st.warning("No data available.")
            return
        
        # Data overview
        st.markdown("### 📊 Data Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2
            st.metric("Memory Usage", f"{memory_usage:.1f} MB")
        
        # Data preview
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        
        # Export options
        st.markdown("### 📥 Export Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name=f"job_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📄 Download JSON",
                data=json_data,
                file_name=f"job_data_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    except Exception as e:
        st.error(f"Error in data management: {str(e)}")

if __name__ == "__main__":
    main() 