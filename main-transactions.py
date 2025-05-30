import logging
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
from selenium import webdriver
from guara.transaction import Application
from guara.transaction import AbstractTransaction
from guara import it
from guara.it import Contains, IsEqualTo

logger = logging.getLogger('rekrute_scraper')

# ----------------------
# Concrete Transactions
# ----------------------
class InitDriver:
    def do(self, headless=False, **kwargs):
        opts = webdriver.ChromeOptions()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        # opts.binary_location = "/usr/bin/chromium"
        self._driver = webdriver.Chrome(options=opts)
        return self._driver

class GetPageUrls(AbstractTransaction):
    def do(self, base_url, **kwargs):
        self._driver.get(base_url)
        wait = WebDriverWait(self._driver, 20)
        select = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "span.jobs select")))
        options = select.find_elements(By.TAG_NAME, "option")
        base_root = base_url.split("/offres.html")[0]
        urls = []
        for opt in options:
            val = opt.get_attribute("value")
            if not val.startswith("http"):
                val = base_root + val.replace("fr/fr/", "fr/")
                urls.append(val)
        return urls

class ScrapeJobListings(AbstractTransaction):
    def do(self, url, **kwargs):
        wait = WebDriverWait(self._driver, 20)
        self._driver.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li.post-id")))
        return self._parse_items()
    
    def _parse_items(self):
        items = self._driver.find_elements(By.CSS_SELECTOR, "li.post-id")
        records = []
        for li in items:
            try:
                records.append(self._parse_job_item(li))
            except Exception as e:
                logger.error(f"Parse error: {e}")
        return records
    
    def _parse_job_item(self, li):
        # ... (same parsing logic as original _scrape_page)
            try:
                records = []
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
            except Exception as e:
                raise e

# ----------------------
# Usage Example
# ----------------------
def test_scraping_flow():
    base_url="https://www.rekrute.com/fr/offres.html"
    db_path: str = "jobs.db"
    max_workers: int = 4
    app = Application(InitDriver().do())
    urls = app.at(GetPageUrls, base_url=base_url).result
    
    # Scrape all pages
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        scrape_page = app.at(ScrapeJobListings, url=base_url).result
        futures = [executor.submit(scrape_page, url) for url in urls]
        return [future.result() for future in as_completed(futures)]
    df = pd.DataFrame(all_records)
    
    # Verify we got some data
    IsEqualTo().validates(len(df) > 0, True)
    
    return df

if __name__ == "__main__":
    test_scraping_flow()