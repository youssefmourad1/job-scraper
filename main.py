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

            driver.get(url)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.post-id")))
            items = driver.find_elements(By.CSS_SELECTOR, "li.post-id")
            logger.info(f"Worker {idx}: found {len(items)} items")
            for li in items:
                try:
                    pid = li.get_attribute("id")
                    sec = li.find_element(By.CSS_SELECTOR, "div.col-sm-10.col-xs-12")
                    title_el = sec.find_element(By.CSS_SELECTOR, "a.titreJob")
                    record = {
                        "post_id": pid,
                        "title": title_el.text.strip(),
                        "url": title_el.get_attribute("href"),
                        "likes": sec.find_element(By.CSS_SELECTOR, "a.addlikebtns span").text.strip(),
                        "company_desc": sec.find_elements(By.CSS_SELECTOR, "div.info span")[0].text.strip() if sec.find_elements(By.CSS_SELECTOR, "div.info span") else "",
                        "mission_desc": sec.find_elements(By.CSS_SELECTOR, "div.info span")[1].text.strip() if len(sec.find_elements(By.CSS_SELECTOR, "div.info span"))>1 else "",
                        "pub_start": "", "pub_end": "", "posts_proposed": "",
                        "sector": "", "fonction": "", "experience": "",
                        "study_level": "", "contract_type": "", "telework": ""
                    }
                    spans = sec.find_elements(By.CSS_SELECTOR, "em.date span")
                    if spans:
                        record['pub_start'] = spans[0].text.strip()
                        if len(spans)>1:
                            record['pub_end'] = spans[1].text.strip()
                        if len(spans)>2:
                            record['posts_proposed'] = spans[2].text.strip()
                    for li2 in sec.find_elements(By.CSS_SELECTOR, "div.info ul li"):
                        txt = li2.text.strip()
                        if txt.startswith("Secteur"):
                            record['sector'] = ", ".join(a.text for a in li2.find_elements(By.TAG_NAME, "a"))
                        elif txt.startswith("Fonction"):
                            record['fonction'] = ", ".join(a.text for a in li2.find_elements(By.TAG_NAME, "a"))
                        elif "Expérience requise" in txt:
                            record['experience'] = li2.find_element(By.TAG_NAME, "a").text.strip()
                        elif "Niveau d'étude" in txt:
                            record['study_level'] = li2.find_element(By.TAG_NAME, "a").text.strip()
                        elif "Type de contrat proposé" in txt:
                            record['contract_type'] = li2.find_element(By.TAG_NAME, "a").text.strip()
                            if "Télétravail" in txt:
                                record['telework'] = txt.split("Télétravail")[-1].split(":")[-1].strip()
                    records.append(record)
                    logger.info(f"Worker {idx}: scraped {pid}")
                except Exception as e:
                    logger.error(f"Worker {idx}: parse error: {e}")
        except TimeoutException:
            logger.error(f"Worker {idx}: timeout on {url}")
        finally:
            driver.quit()
        return records

    def scrape_all(self, show_logs_callback=None):
        urls = self.get_page_urls()
        total = len(urls)
        records = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {executor.submit(self._scrape_page, url, i+1, total): i for i, url in enumerate(urls)}
            for future in as_completed(future_to_idx):
                page_records = future.result()
                records.extend(page_records)
                if show_logs_callback:
                    show_logs_callback(log_stream.getvalue())
        return pd.DataFrame(records)

    def save_to_db(self, df: pd.DataFrame):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs(
                post_id TEXT PRIMARY KEY,
                title TEXT, url TEXT, likes TEXT,
                company_desc TEXT, mission_desc TEXT,
                pub_start TEXT, pub_end TEXT, posts_proposed TEXT,
                sector TEXT, fonction TEXT, experience TEXT,
                study_level TEXT, contract_type TEXT, telework TEXT
            )"""
        )
        existing = {row[0] for row in c.execute("SELECT post_id FROM jobs").fetchall()}
        new_df = df[~df['post_id'].isin(existing)]
        new_count = len(new_df)
        if new_count:
            new_df.to_sql('jobs', conn, if_exists='append', index=False)
        conn.close()
        logger.info(f"Inserted {new_count} new records")
        return new_count

# ----------------------
# Streamlit App
# ----------------------
st.title("Rekrute.com Job Scraper")
DB_PATH = "jobs.db"
scraper = RekruteScraper( )
import os 
# Check for existing data
has_data = False
if os.path.exists(DB_PATH):
    try:
        df_db = pd.read_sql("SELECT * FROM jobs", sqlite3.connect(DB_PATH))
        has_data = not df_db.empty
    except:
        has_data = False

# Tabs setup
if has_data:
    tab1, tab2 = st.tabs(["Launch Scrape", "Data & Dashboard"])
else:
    tab1 = st.container()

# Scraping UI
with tab1:
    scraper.base_url = st.text_input("Base URL", scraper.base_url)
    log_placeholder = st.empty()
    if st.button("Launch Scraping"):
        log_stream.truncate(0)
        log_stream.seek(0)
        def update_logs(txt):
            log_placeholder.text(txt)
        with st.spinner("Scraping pages in parallel…"):
            df = scraper.scrape_all(show_logs_callback=update_logs)
            new = scraper.save_to_db(df)
        st.success(f"Done: {new} new records")
#fhjlkfd

if has_data:
    with tab2:
        st.subheader("Offers")
        st.dataframe(df_db)
        st.download_button("Download DB",
                           data=open(DB_PATH, 'rb'),
                           file_name="jobs.db")
        st.subheader("Dashboard")
        st.metric("Total Offers", len(df_db))
        sector_chart = alt.Chart(df_db).mark_bar().encode(
            x=alt.X('sector', sort='-y'), y='count()'
        )
        st.altair_chart(sector_chart, use_container_width=True)
        func_chart = alt.Chart(df_db).mark_bar().encode(
            x=alt.X('fonction', sort='-y'), y='count()'
        )
        st.altair_chart(func_chart, use_container_width=True)

