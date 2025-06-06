"""
Database management for Job Market Analyzer
"""
import sqlite3
import pandas as pd
from contextlib import contextmanager
from datetime import datetime
import hashlib
import logging
from typing import List, Dict, Optional
from .settings import DATABASE_CONFIG

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Centralized database management with deduplication and optimization
    """
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DATABASE_CONFIG['main_db']
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize database with optimized schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Main jobs table with comprehensive schema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_hash TEXT UNIQUE NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    company TEXT,
                    location TEXT,
                    sector TEXT,
                    fonction TEXT,
                    experience TEXT,
                    education_level TEXT,
                    contract_type TEXT,
                    salary TEXT,
                    salary_min INTEGER,
                    salary_max INTEGER,
                    description TEXT,
                    requirements TEXT,
                    url TEXT UNIQUE,
                    date_posted DATE,
                    date_scraped TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    view_count INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Skills table for normalized skill tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT UNIQUE NOT NULL,
                    category TEXT,
                    frequency INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Job-Skills relationship table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS job_skills (
                    job_id INTEGER,
                    skill_id INTEGER,
                    relevance_score REAL DEFAULT 1.0,
                    PRIMARY KEY (job_id, skill_id),
                    FOREIGN KEY (job_id) REFERENCES jobs(id),
                    FOREIGN KEY (skill_id) REFERENCES skills(id)
                )
            """)
            
            # Market trends table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    sector TEXT,
                    location TEXT,
                    job_count INTEGER,
                    avg_salary REAL,
                    growth_rate REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, sector, location)
                )
            """)
            
            # Scraping logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scraping_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    jobs_scraped INTEGER DEFAULT 0,
                    jobs_new INTEGER DEFAULT 0,
                    jobs_updated INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source)",
                "CREATE INDEX IF NOT EXISTS idx_jobs_sector ON jobs(sector)",
                "CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location)",
                "CREATE INDEX IF NOT EXISTS idx_jobs_date_posted ON jobs(date_posted)",
                "CREATE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(job_hash)",
                "CREATE INDEX IF NOT EXISTS idx_market_trends_date ON market_trends(date)",
                "CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(skill_name)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def generate_job_hash(self, job_data: Dict) -> str:
        """Generate unique hash for job deduplication"""
        # Use title, company, and url for unique identification
        unique_string = f"{job_data.get('title', '')}-{job_data.get('company', '')}-{job_data.get('url', '')}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def insert_job(self, job_data: Dict) -> bool:
        """Insert job with deduplication check"""
        job_hash = self.generate_job_hash(job_data)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if job already exists
            cursor.execute("SELECT id FROM jobs WHERE job_hash = ?", (job_hash,))
            existing_job = cursor.fetchone()
            
            if existing_job:
                # Update existing job
                cursor.execute("""
                    UPDATE jobs SET 
                        updated_at = CURRENT_TIMESTAMP,
                        view_count = ?,
                        likes = ?,
                        is_active = 1
                    WHERE job_hash = ?
                """, (
                    job_data.get('view_count', 0),
                    job_data.get('likes', 0),
                    job_hash
                ))
                logger.debug(f"Updated existing job: {job_data.get('title', 'Unknown')}")
                return False  # Not a new job
            else:
                # Insert new job
                cursor.execute("""
                    INSERT INTO jobs (
                        job_hash, source, title, company, location, sector, fonction,
                        experience, education_level, contract_type, salary, salary_min,
                        salary_max, description, requirements, url, date_posted,
                        view_count, likes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_hash,
                    job_data.get('source', ''),
                    job_data.get('title', ''),
                    job_data.get('company', ''),
                    job_data.get('location', ''),
                    job_data.get('sector', ''),
                    job_data.get('fonction', ''),
                    job_data.get('experience', ''),
                    job_data.get('education_level', ''),
                    job_data.get('contract_type', ''),
                    job_data.get('salary', ''),
                    job_data.get('salary_min'),
                    job_data.get('salary_max'),
                    job_data.get('description', ''),
                    job_data.get('requirements', ''),
                    job_data.get('url', ''),
                    job_data.get('date_posted'),
                    job_data.get('view_count', 0),
                    job_data.get('likes', 0)
                ))
                conn.commit()
                logger.debug(f"Inserted new job: {job_data.get('title', 'Unknown')}")
                return True  # New job inserted
    
    def bulk_insert_jobs(self, jobs_data: List[Dict]) -> Dict[str, int]:
        """Bulk insert jobs with statistics"""
        stats = {'new': 0, 'updated': 0, 'total': len(jobs_data)}
        
        for job_data in jobs_data:
            if self.insert_job(job_data):
                stats['new'] += 1
            else:
                stats['updated'] += 1
        
        return stats
    
    def get_jobs_dataframe(self, filters: Dict = None) -> pd.DataFrame:
        """Get jobs as pandas DataFrame with optional filters"""
        query = "SELECT * FROM jobs WHERE is_active = 1"
        params = []
        
        if filters:
            if 'source' in filters:
                query += " AND source = ?"
                params.append(filters['source'])
            if 'sector' in filters:
                query += " AND sector LIKE ?"
                params.append(f"%{filters['sector']}%")
            if 'location' in filters:
                query += " AND location LIKE ?"
                params.append(f"%{filters['location']}%")
            if 'date_from' in filters:
                query += " AND date_posted >= ?"
                params.append(filters['date_from'])
            if 'date_to' in filters:
                query += " AND date_posted <= ?"
                params.append(filters['date_to'])
        
        query += " ORDER BY date_posted DESC"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def get_market_stats(self) -> Dict:
        """Get comprehensive market statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Basic stats
            cursor.execute("SELECT COUNT(*) FROM jobs WHERE is_active = 1")
            total_jobs = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT company) FROM jobs WHERE is_active = 1")
            unique_companies = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT sector) FROM jobs WHERE is_active = 1")
            unique_sectors = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT location) FROM jobs WHERE is_active = 1")
            unique_locations = cursor.fetchone()[0]
            
            # Recent activity
            cursor.execute("""
                SELECT COUNT(*) FROM jobs 
                WHERE is_active = 1 AND date_scraped >= date('now', '-7 days')
            """)
            jobs_last_week = cursor.fetchone()[0]
            
            return {
                'total_jobs': total_jobs,
                'unique_companies': unique_companies,
                'unique_sectors': unique_sectors,
                'unique_locations': unique_locations,
                'jobs_last_week': jobs_last_week
            }
    
    def log_scraping_session(self, source: str, stats: Dict) -> int:
        """Log scraping session with statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scraping_logs (
                    source, start_time, end_time, jobs_scraped, 
                    jobs_new, jobs_updated, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                source,
                stats.get('start_time'),
                stats.get('end_time'),
                stats.get('total', 0),
                stats.get('new', 0),
                stats.get('updated', 0),
                'completed'
            ))
            conn.commit()
            return cursor.lastrowid
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old inactive jobs"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM jobs 
                WHERE is_active = 0 AND updated_at < date('now', '-{} days')
            """.format(days))
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"Cleaned up {deleted_count} old job records")
            return deleted_count 