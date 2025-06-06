"""
Base scraper class with optimized parallel processing
"""
import asyncio
import aiohttp
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Optional, Callable
import logging
from dataclasses import dataclass

from config.database import DatabaseManager
from config.settings import SCRAPER_CONFIG
from utils.helpers import get_random_user_agent, clean_text, parse_date, extract_salary_range

logger = logging.getLogger('scrapers')

@dataclass
class ScrapingSession:
    """Data class to track scraping session information"""
    source: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_jobs: int = 0
    new_jobs: int = 0
    updated_jobs: int = 0
    errors: int = 0
    status: str = 'running'

class BaseScraper(ABC):
    """
    Abstract base class for all job scrapers with optimized parallel processing
    """
    
    def __init__(self, source_name: str, base_url: str):
        self.source_name = source_name
        self.base_url = base_url
        self.db_manager = DatabaseManager()
        self.session = ScrapingSession(source=source_name, start_time=datetime.now())
        
        # Configuration
        self.max_workers = SCRAPER_CONFIG['max_workers']
        self.request_delay = SCRAPER_CONFIG['request_delay']
        self.timeout = SCRAPER_CONFIG['timeout']
        self.retry_attempts = SCRAPER_CONFIG['retry_attempts']
        
        logger.info(f"Initialized {source_name} scraper")
    
    @abstractmethod
    async def get_page_urls(self) -> List[str]:
        """Get list of pages to scrape"""
        pass
    
    @abstractmethod
    async def scrape_job_listings(self, page_url: str) -> List[Dict]:
        """Scrape job listings from a single page"""
        pass
    
    @abstractmethod
    async def scrape_job_details(self, job_basic_info: Dict) -> Dict:
        """Scrape detailed information for a single job"""
        pass
    
    async def make_request(self, session: aiohttp.ClientSession, url: str, **kwargs) -> Optional[str]:
        """Make HTTP request with retry logic and rate limiting"""
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        for attempt in range(self.retry_attempts):
            try:
                await asyncio.sleep(self.request_delay)  # Rate limiting
                
                async with session.get(
                    url, 
                    headers=headers, 
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    **kwargs
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        logger.debug(f"Successfully fetched: {url}")
                        return content
                    elif response.status in [404, 403, 410]:
                        # Don't retry client errors (not found, forbidden, gone)
                        logger.debug(f"HTTP {response.status} for {url} - skipping (permanent error)")
                        return None
                    elif response.status in [500, 502, 503, 504]:
                        # Retry server errors
                        logger.warning(f"HTTP {response.status} for {url} (attempt {attempt + 1}) - server error, will retry")
                    else:
                        # Other errors - log and potentially retry
                        logger.warning(f"HTTP {response.status} for {url} (attempt {attempt + 1})")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url} (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Error fetching {url} (attempt {attempt + 1}): {str(e)}")
            
            # Don't retry if it's the last attempt
            if attempt < self.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        self.session.errors += 1
        return None
    
    def process_job_data(self, raw_job_data: Dict) -> Dict:
        """Process and clean job data before database insertion"""
        processed_data = {
            'source': self.source_name,
            'title': clean_text(raw_job_data.get('title', '')),
            'company': clean_text(raw_job_data.get('company', '')),
            'location': clean_text(raw_job_data.get('location', '')),
            'sector': clean_text(raw_job_data.get('sector', '')),
            'fonction': clean_text(raw_job_data.get('fonction', '')),
            'experience': clean_text(raw_job_data.get('experience', '')),
            'education_level': clean_text(raw_job_data.get('education_level', '')),
            'contract_type': clean_text(raw_job_data.get('contract_type', '')),
            'description': clean_text(raw_job_data.get('description', '')),
            'requirements': clean_text(raw_job_data.get('requirements', '')),
            'url': raw_job_data.get('url', ''),
            'view_count': int(raw_job_data.get('view_count', 0)) if raw_job_data.get('view_count') else 0,
            'likes': int(raw_job_data.get('likes', 0)) if raw_job_data.get('likes') else 0,
        }
        
        # Parse salary information
        salary_str = raw_job_data.get('salary', '')
        processed_data['salary'] = salary_str
        salary_range = extract_salary_range(salary_str)
        processed_data['salary_min'] = salary_range['min']
        processed_data['salary_max'] = salary_range['max']
        
        # Parse date
        date_posted = parse_date(raw_job_data.get('date_posted', ''))
        processed_data['date_posted'] = date_posted.strftime('%Y-%m-%d') if date_posted else None
        
        return processed_data
    
    async def scrape_page_parallel(self, session: aiohttp.ClientSession, page_url: str) -> List[Dict]:
        """Scrape a single page and return processed job data"""
        try:
            # Get basic job listings from page
            job_listings = await self.scrape_job_listings(page_url)
            
            if not job_listings:
                return []
            
            # Scrape detailed information for each job
            detailed_jobs = []
            for job_basic in job_listings:
                try:
                    job_details = await self.scrape_job_details(job_basic)
                    if job_details:
                        processed_job = self.process_job_data(job_details)
                        detailed_jobs.append(processed_job)
                        logger.debug(f"Processed job: {processed_job.get('title', 'Unknown')}")
                except Exception as e:
                    logger.error(f"Error processing job details: {str(e)}")
                    self.session.errors += 1
            
            return detailed_jobs
            
        except Exception as e:
            logger.error(f"Error scraping page {page_url}: {str(e)}")
            self.session.errors += 1
            return []
    
    async def run_scraping_session(self, max_pages: Optional[int] = None, 
                                 progress_callback: Optional[Callable] = None) -> ScrapingSession:
        """Run complete scraping session with parallel processing"""
        logger.info(f"Starting scraping session for {self.source_name}")
        
        try:
            # Get all page URLs
            page_urls = await self.get_page_urls()
            if max_pages:
                page_urls = page_urls[:max_pages]
            
            logger.info(f"Found {len(page_urls)} pages to scrape")
            
            # Create aiohttp session with connection pooling
            connector = aiohttp.TCPConnector(limit=self.max_workers, limit_per_host=self.max_workers)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                # Process pages in parallel with controlled concurrency
                semaphore = asyncio.Semaphore(self.max_workers)
                
                async def scrape_with_semaphore(page_url):
                    async with semaphore:
                        return await self.scrape_page_parallel(session, page_url)
                
                # Execute scraping tasks
                tasks = [scrape_with_semaphore(url) for url in page_urls]
                
                all_jobs = []
                completed = 0
                
                for coro in asyncio.as_completed(tasks):
                    jobs = await coro
                    all_jobs.extend(jobs)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(page_urls), len(all_jobs))
                    
                    logger.info(f"Completed {completed}/{len(page_urls)} pages, found {len(jobs)} jobs")
            
            # Save to database
            if all_jobs:
                db_stats = self.db_manager.bulk_insert_jobs(all_jobs)
                self.session.total_jobs = db_stats['total']
                self.session.new_jobs = db_stats['new']
                self.session.updated_jobs = db_stats['updated']
                
                logger.info(f"Database stats: {db_stats}")
            
            # Finalize session
            self.session.end_time = datetime.now()
            self.session.status = 'completed'
            
            # Log session to database
            session_stats = {
                'start_time': self.session.start_time,
                'end_time': self.session.end_time,
                'total': self.session.total_jobs,
                'new': self.session.new_jobs,
                'updated': self.session.updated_jobs
            }
            self.db_manager.log_scraping_session(self.source_name, session_stats)
            
            logger.info(f"Completed scraping session for {self.source_name}: "
                       f"{self.session.new_jobs} new jobs, "
                       f"{self.session.updated_jobs} updated jobs, "
                       f"{self.session.errors} errors")
            
        except Exception as e:
            logger.error(f"Scraping session failed for {self.source_name}: {str(e)}")
            self.session.end_time = datetime.now()
            self.session.status = 'failed'
        
        return self.session
    
    def run_sync(self, max_pages: Optional[int] = None, 
                progress_callback: Optional[Callable] = None) -> ScrapingSession:
        """Synchronous wrapper for async scraping"""
        return asyncio.run(self.run_scraping_session(max_pages, progress_callback)) 