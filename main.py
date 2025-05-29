import streamlit as st
import pandas as pd
import sqlite3
import os
import logging
from io import StringIO
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import altair as alt
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------
# Logger Setup
# ----------------------
log_stream = StringIO()
handler = logging.StreamHandler(log_stream)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger = logging.getLogger('rekrute_scraper')
logger.setLevel(logging.INFO)
logger.addHandler(handler)
# Also log to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ----------------------
# Scraper Class
# ----------------------
class RekruteScraper:
    def __init__(self, base_url: str, db_path: str = "jobs.db", max_workers: int = 4):
        self.base_url = base_url
        self.db_path = db_path
        self.max_workers = max_workers


    def _init_driver(self, headless: bool = True):
        chrome_opts = Options()

        # --- Essential options for headless/containerized environments ---
        if headless:
            chrome_opts.add_argument("--headless=new")  # Use the new headless mode
            chrome_opts.add_argument("--disable-gpu")   # Often recommended for headless
        
        chrome_opts.add_argument("--no-sandbox")        # Crucial for running in containers like Streamlit Cloud
        chrome_opts.add_argument("--disable-dev-shm-usage") # Overcomes limited resource problems in containers
        
        # --- Optional but often useful options ---
        chrome_opts.add_argument("--window-size=1920,1080") # Define a virtual display size
        chrome_opts.add_argument("--disable-blink-features=AutomationControlled") # Helps avoid bot detection
        chrome_opts.add_argument("--start-maximized") # Could be useful
        chrome_opts.add_argument("--disable-extensions") # Disable extensions
        chrome_opts.add_argument("--disable-infobars") # Disable "Chrome is being controlled by..."
        chrome_opts.add_argument("--remote-debugging-port=9222") # Can be useful for debugging headless if needed

        # --- Point to the browser installed by packages.txt ---
        # This tells Selenium/WebDriverManager which browser executable to consider.
        # Adjust the path if you installed a different package (e.g., google-chrome-stable)
        # Common paths for browsers installed via apt:
        browser_executable_path = ""
        common_paths = ["/usr/bin/chromium-browser", "/usr/bin/chromium", "/usr/bin/google-chrome-stable", "/usr/bin/google-chrome"]
        
        import os
        for path in common_paths:
            if os.path.exists(path):
                browser_executable_path = path
                break
                
        if browser_executable_path:
            chrome_opts.binary_location = browser_executable_path
            # print(f"INFO: Using browser binary at: {browser_executable_path}") # For debugging
        else:
            # print("WARNING: Browser executable not found at common paths. Relying on WebDriver to find it.")
            # This is less ideal, as WebDriverManager might not pick the intended browser
            pass

        # --- Initialize ChromeDriver using WebDriverManager ---
        # WebDriverManager will attempt to download a chromedriver compatible with
        # the browser found (ideally the one specified by binary_location or in PATH).
        # Since packages.txt should have installed all necessary shared libraries,
        # the downloaded chromedriver should now run without the 'exit code 127' error.
        try:
            service = Service(ChromeDriverManager().install())
        except Exception as e:
            # Handle potential errors from ChromeDriverManager (e.g., network issues during download)
            # print(f"ERROR: ChromeDriverManager().install() failed: {e}")
            # As a last resort, if you also included 'chromium-driver' in packages.txt,
            # you could try to use the system-installed chromedriver.
            # This assumes 'chromium-driver' from apt is compatible with 'chromium-browser'.
            chromedriver_sys_path = "/usr/bin/chromedriver" # Common path for system chromedriver
            if os.path.exists(chromedriver_sys_path) and browser_executable_path: # Ensure browser is also set
                # print(f"INFO: Falling back to system chromedriver at: {chromedriver_sys_path}")
                service = Service(executable_path=chromedriver_sys_path)
            else:
                # print(f"ERROR: System chromedriver not found at {chromedriver_sys_path} or browser binary not set. Cannot initialize driver.")
                raise  # Re-raise the original exception from ChromeDriverManager

        return webdriver.Chrome(service=service, options=chrome_opts)

    def get_page_urls(self):
        logger.info("Fetching pagination URLs")
        driver = self._init_driver()
        driver.get(self.base_url)
        wait = WebDriverWait(driver, 20)
        select = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.jobs select")))
        options = select.find_elements(By.TAG_NAME, "option")
        base_root = self.base_url.split("/offres.html")[0]
        urls = []
        for opt in options:
            val = opt.get_attribute("value")
            if not val.startswith("http"):
                val = base_root + val
                # Clean the URLs from repeating "fr/"
                val = val.replace("fr/fr/", "fr/")
                urls.append(val)
                logger.info(f"Found page: {val}")

        logger.info(f"Total pages found: {len(urls)}")
        logger.info(urls )
        driver.quit()
        logger.info(f"Total pages: {len(urls)}")
        return urls

    def _scrape_page(self, url: str, idx: int, total: int):
        driver = self._init_driver()
        wait = WebDriverWait(driver, 20)
        logger.info(f"Worker {idx}/{total} starting {url}")
        records = []
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
scraper = RekruteScraper(base_url="https://www.rekrute.com/fr/offres.html?s=3&p=1&o=1",
                        db_path=DB_PATH, max_workers=5)

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

# Data & Dashboard
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
