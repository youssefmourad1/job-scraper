"""
Optimized MarocAnnonce scraper with async processing
"""
import re
import aiohttp
from typing import List, Dict
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging

from .base_scraper import BaseScraper
from config.settings import SCRAPER_URLS
from utils.helpers import clean_text, validate_url

logger = logging.getLogger('scrapers.maroc_annonce')

class MarocAnnonceScraper(BaseScraper):
    """
    Optimized MarocAnnonce.com scraper with parallel processing
    """
    
    def __init__(self):
        base_url = SCRAPER_URLS['maroc_annonce']['base_url']
        super().__init__('MarocAnnonce', base_url)
        self.search_params = SCRAPER_URLS['maroc_annonce']['search_params']
    
    async def get_page_urls(self) -> List[str]:
        """Get all pagination URLs"""
        page_urls = []
        max_pages = 100  # Reasonable limit
        
        for page_num in range(1, max_pages + 1):
            page_url = f"{self.base_url}?pge={page_num}"
            page_urls.append(page_url)
        
        logger.info(f"Generated {len(page_urls)} page URLs for MarocAnnonce")
        return page_urls
    
    async def scrape_job_listings(self, page_url: str) -> List[Dict]:
        """Scrape job listings from a single page"""
        async with aiohttp.ClientSession() as session:
            content = await self.make_request(session, page_url)
            
            if not content:
                return []
            
            soup = BeautifulSoup(content, 'html.parser')
            job_listings = []
            
            # Find the cars-list ul element
            cars_list = soup.find('ul', class_='cars-list')
            if not cars_list:
                return []
            
            # Extract job items (excluding ads)
            job_items = cars_list.find_all('li')
            
            for item in job_items:
                try:
                    # Skip ad items
                    if item.find('div', class_='ad-item'):
                        continue
                    
                    job_data = self._extract_job_basic_info(item)
                    if job_data and job_data.get('title'):
                        job_listings.append(job_data)
                except Exception as e:
                    logger.error(f"Error extracting job from item: {str(e)}")
            
            logger.debug(f"Found {len(job_listings)} jobs on page: {page_url}")
            return job_listings
    
    def _extract_job_basic_info(self, item) -> Dict:
        """Extract basic job information from a job listing item"""
        job_data = {}
        
        # Extract ad ID if available
        if item.get('id'):
            job_data['ad_id'] = item.get('id')
        
        # Find title and URL
        title_link = item.find('a', href=True)
        if title_link:
            job_data['title'] = clean_text(title_link.get_text())
            job_data['url'] = urljoin('https://www.marocannonces.com', title_link['href'])
        
        # Extract company info
        company_div = item.find('div', class_='company-name')
        if company_div:
            job_data['company'] = clean_text(company_div.get_text())
        
        # Extract location
        location_span = item.find('span', class_='location')
        if location_span:
            job_data['location'] = clean_text(location_span.get_text())
        
        # Extract date posted
        date_span = item.find('span', class_='date')
        if date_span:
            job_data['date_posted'] = clean_text(date_span.get_text())
        
        # Extract time posted
        time_span = item.find('span', class_='time')
        if time_span:
            job_data['time_posted'] = clean_text(time_span.get_text())
        
        return job_data
    
    async def scrape_job_details(self, job_basic_info: Dict) -> Dict:
        """Scrape detailed information for a single job"""
        job_url = job_basic_info.get('url')
        if not job_url or not validate_url(job_url):
            logger.debug(f"Invalid or missing URL for job: {job_basic_info.get('title', 'Unknown')}")
            return job_basic_info
        
        async with aiohttp.ClientSession() as session:
            content = await self.make_request(session, job_url)
            
            if not content:
                # Job detail page not accessible (404, etc.) - return basic info
                logger.debug(f"Could not fetch details for job: {job_basic_info.get('title', 'Unknown')} - using basic info only")
                return job_basic_info
            
            try:
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract description
                description_div = soup.find('div', class_='description')
                if description_div:
                    # Remove script tags and clean text
                    for script in description_div.find_all('script'):
                        script.decompose()
                    description = description_div.get_text().strip()
                    job_basic_info['description'] = clean_text(description)
                    
                    # Parse additional details from description
                    parsed_details = self._parse_description_details(description)
                    job_basic_info.update(parsed_details)
                
                # Extract additional parameters
                self._extract_extra_questions(soup, job_basic_info)
                
                # Extract advertiser info
                self._extract_advertiser_info(soup, job_basic_info)
                
                # Extract view count
                self._extract_view_count(soup, job_basic_info)
                
            except Exception as e:
                logger.warning(f"Error parsing job details for {job_basic_info.get('title', 'Unknown')}: {str(e)}")
            
            return job_basic_info
    
    def _parse_description_details(self, description: str) -> Dict:
        """Parse structured information from job description"""
        parsed_data = {}
        
        if not description:
            return parsed_data
        
        # Extract phone number
        phone_match = re.search(r'(\d{3}-\d\*+|\d{10,})', description)
        if phone_match:
            parsed_data['phone_number'] = phone_match.group(1)
        
        # Extract missions/responsibilities
        missions_match = re.search(
            r'Missions principales?\s*:(.+?)(?=Profil recherché|Conditions|$)', 
            description, 
            re.DOTALL | re.IGNORECASE
        )
        if missions_match:
            missions = missions_match.group(1).strip()
            missions = re.sub(r'\s+', ' ', missions)
            parsed_data['job_missions'] = missions
        
        # Extract required profile
        profile_match = re.search(
            r'Profil recherché\s*:(.+?)(?=Conditions|Expérience|$)', 
            description, 
            re.DOTALL | re.IGNORECASE
        )
        if profile_match:
            profile = profile_match.group(1).strip()
            profile = re.sub(r'\s+', ' ', profile)
            parsed_data['requirements'] = profile
        
        # Extract work conditions
        conditions_match = re.search(
            r'Conditions?\s*:(.+?)(?=Domaine|Annonceur|$)', 
            description, 
            re.DOTALL | re.IGNORECASE
        )
        if conditions_match:
            conditions = conditions_match.group(1).strip()
            conditions = re.sub(r'\s+', ' ', conditions)
            parsed_data['work_conditions'] = conditions
        
        # Extract city from structured data
        city_match = re.search(r'Ville\s*:\s*(.+?)(?=\n|$)', description)
        if city_match:
            parsed_data['city'] = city_match.group(1).strip()
        
        return parsed_data
    
    def _extract_extra_questions(self, soup: BeautifulSoup, job_data: Dict):
        """Extract additional parameters from extra questions section"""
        extra_questions = soup.find('ul', class_='extraQuestionName')
        if extra_questions:
            for li in extra_questions.find_all('li'):
                text = li.get_text().strip()
                if text.startswith('Domaine :'):
                    job_data['sector'] = text.replace('Domaine :', '').strip()
                elif text.startswith('Fonction :'):
                    job_data['fonction'] = text.replace('Fonction :', '').strip()
                elif text.startswith('Contrat :'):
                    job_data['contract_type'] = text.replace('Contrat :', '').strip()
                elif text.startswith('Entreprise :'):
                    job_data['company_type'] = text.replace('Entreprise :', '').strip()
                elif text.startswith('Salaire :'):
                    job_data['salary'] = text.replace('Salaire :', '').strip()
                elif text.startswith('Niveau d\'études :'):
                    job_data['education_level'] = text.replace('Niveau d\'études :', '').strip()
    
    def _extract_advertiser_info(self, soup: BeautifulSoup, job_data: Dict):
        """Extract advertiser information"""
        infoannonce = soup.find('div', class_='infoannonce')
        if infoannonce:
            dt_tags = infoannonce.find_all('dt')
            for dt in dt_tags:
                if 'Annonceur' in dt.text:
                    dd = dt.find_next_sibling('dd')
                    if dd:
                        job_data['advertiser'] = dd.text.strip()
    
    def _extract_view_count(self, soup: BeautifulSoup, job_data: Dict):
        """Extract view count information"""
        info_holder = soup.find('ul', class_='info-holder')
        if info_holder:
            for li in info_holder.find_all('li'):
                text = li.get_text().strip()
                if 'Vue:' in text:
                    match = re.search(r'Vue:\s*(\d+)', text)
                    if match:
                        job_data['view_count'] = int(match.group(1)) 