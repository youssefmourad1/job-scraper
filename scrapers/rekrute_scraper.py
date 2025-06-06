"""
Optimized Rekrute scraper with async processing
"""
import re
import aiohttp
from typing import List, Dict
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, parse_qs
import logging

from .base_scraper import BaseScraper
from config.settings import SCRAPER_URLS
from utils.helpers import clean_text, validate_url

logger = logging.getLogger('scrapers.rekrute')

class RekruteScraper(BaseScraper):
    """
    Optimized Rekrute.com scraper with parallel processing
    """
    
    def __init__(self):
        base_url = SCRAPER_URLS['rekrute']['base_url']
        super().__init__('Rekrute', base_url)
        self.search_params = SCRAPER_URLS['rekrute']['search_params']
    
    async def get_page_urls(self) -> List[str]:
        """Get all pagination URLs from the main page"""
        async with self.db_manager.get_connection() as conn:
            async with aiohttp.ClientSession() as session:
                main_page_content = await self.make_request(session, self.base_url)
                
                if not main_page_content:
                    logger.error("Failed to fetch main page")
                    return []
                
                soup = BeautifulSoup(main_page_content, 'html.parser')
                page_urls = []
                
                # Find pagination container
                pagination = soup.find('span', class_='jobs')
                if pagination:
                    select_element = pagination.find('select')
                    if select_element:
                        options = select_element.find_all('option')
                        
                        for option in options:
                            value = option.get('value', '')
                            if value and value.startswith('/'):
                                # Construct full URL
                                full_url = urljoin(self.base_url, value)
                                # Clean up any duplicate path segments
                                full_url = re.sub(r'/fr/fr/', '/fr/', full_url)
                                
                                if validate_url(full_url):
                                    page_urls.append(full_url)
                                    logger.debug(f"Found page URL: {full_url}")
                
                logger.info(f"Found {len(page_urls)} page URLs for Rekrute")
                return page_urls or [self.base_url]  # Return at least the main page
    
    async def scrape_job_listings(self, page_url: str) -> List[Dict]:
        """Scrape job listings from a single page"""
        async with aiohttp.ClientSession() as session:
            content = await self.make_request(session, page_url)
            
            if not content:
                return []
            
            soup = BeautifulSoup(content, 'html.parser')
            job_listings = []
            
            # Find job listings
            job_items = soup.find_all('li', class_='post-id')
            
            for item in job_items:
                try:
                    job_data = self._extract_job_basic_info(item)
                    if job_data:
                        job_listings.append(job_data)
                except Exception as e:
                    logger.error(f"Error extracting job from item: {str(e)}")
            
            logger.debug(f"Found {len(job_listings)} jobs on page: {page_url}")
            return job_listings
    
    def _extract_job_basic_info(self, item) -> Dict:
        """Extract basic job information from a job listing item"""
        job_data = {}
        
        # Extract post ID
        job_data['post_id'] = item.get('id', '')
        
        # Find the main content section
        content_section = item.find('div', class_='col-sm-10')
        if not content_section:
            return {}
        
        # Extract title and URL
        title_link = content_section.find('a', class_='titreJob')
        if title_link:
            job_data['title'] = clean_text(title_link.get_text())
            job_data['url'] = title_link.get('href', '')
            
            # Ensure URL is absolute
            if job_data['url'] and not job_data['url'].startswith('http'):
                job_data['url'] = urljoin(self.base_url, job_data['url'])
        
        # Extract likes
        likes_element = content_section.find('a', class_='addlikebtns')
        if likes_element:
            likes_span = likes_element.find('span')
            if likes_span:
                job_data['likes'] = self._extract_number(likes_span.get_text())
        
        # Extract company and mission description
        info_spans = content_section.find_all('div', class_='info')
        if info_spans:
            spans = info_spans[0].find_all('span')
            if len(spans) > 0:
                job_data['company'] = clean_text(spans[0].get_text())
            if len(spans) > 1:
                job_data['mission_desc'] = clean_text(spans[1].get_text())
        
        # Extract dates and posts count
        date_section = content_section.find('em', class_='date')
        if date_section:
            date_spans = date_section.find_all('span')
            if len(date_spans) > 0:
                job_data['date_posted'] = clean_text(date_spans[0].get_text())
            if len(date_spans) > 1:
                job_data['pub_end'] = clean_text(date_spans[1].get_text())
            if len(date_spans) > 2:
                job_data['posts_proposed'] = clean_text(date_spans[2].get_text())
        
        # Extract job details from info list
        info_lists = content_section.find_all('div', class_='info')
        for info_div in info_lists:
            ul_element = info_div.find('ul')
            if ul_element:
                for li in ul_element.find_all('li'):
                    self._extract_job_detail(li, job_data)
        
        return job_data
    
    def _extract_job_detail(self, li_element, job_data: Dict):
        """Extract specific job details from list items"""
        text = li_element.get_text().strip()
        
        if text.startswith('Secteur'):
            links = li_element.find_all('a')
            job_data['sector'] = ', '.join([clean_text(a.get_text()) for a in links])
        
        elif text.startswith('Fonction'):
            links = li_element.find_all('a')
            job_data['fonction'] = ', '.join([clean_text(a.get_text()) for a in links])
        
        elif 'Expérience requise' in text:
            link = li_element.find('a')
            if link:
                job_data['experience'] = clean_text(link.get_text())
        
        elif 'Niveau d\'étude' in text:
            link = li_element.find('a')
            if link:
                job_data['education_level'] = clean_text(link.get_text())
        
        elif 'Type de contrat proposé' in text:
            link = li_element.find('a')
            if link:
                job_data['contract_type'] = clean_text(link.get_text())
            
            # Check for telework information
            if 'Télétravail' in text:
                telework_part = text.split('Télétravail')[-1]
                job_data['telework'] = clean_text(telework_part.split(':')[-1])
        
        elif 'Lieu de travail' in text or 'Localisation' in text:
            link = li_element.find('a')
            if link:
                job_data['location'] = clean_text(link.get_text())
    
    async def scrape_job_details(self, job_basic_info: Dict) -> Dict:
        """Scrape detailed information for a single job"""
        job_url = job_basic_info.get('url')
        if not job_url or not validate_url(job_url):
            return job_basic_info
        
        async with aiohttp.ClientSession() as session:
            content = await self.make_request(session, job_url)
            
            if not content:
                return job_basic_info
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract detailed description
            description_div = soup.find('div', class_='job-description')
            if description_div:
                job_basic_info['description'] = clean_text(description_div.get_text())
            
            # Extract requirements if available
            requirements_section = soup.find('div', class_='requirements')
            if requirements_section:
                job_basic_info['requirements'] = clean_text(requirements_section.get_text())
            
            # Extract additional details from the job page
            self._extract_additional_details(soup, job_basic_info)
            
            return job_basic_info
    
    def _extract_additional_details(self, soup: BeautifulSoup, job_data: Dict):
        """Extract additional details from the job detail page"""
        # Look for salary information
        salary_elements = soup.find_all(text=re.compile(r'salaire|rémunération|MAD|DH', re.IGNORECASE))
        for element in salary_elements:
            parent = element.parent
            if parent:
                salary_text = clean_text(parent.get_text())
                if any(keyword in salary_text.lower() for keyword in ['salaire', 'rémunération']):
                    job_data['salary'] = salary_text
                    break
        
        # Extract view count if available
        view_elements = soup.find_all(text=re.compile(r'vue|vu|vues', re.IGNORECASE))
        for element in view_elements:
            view_text = element.strip()
            view_number = self._extract_number(view_text)
            if view_number:
                job_data['view_count'] = view_number
                break
    
    def _extract_number(self, text: str) -> int:
        """Extract first number from text"""
        if not text:
            return 0
        
        numbers = re.findall(r'\d+', str(text))
        return int(numbers[0]) if numbers else 0 