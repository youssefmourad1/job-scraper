"""
Logging configuration for Job Market Analyzer
"""
import logging
import logging.handlers
from pathlib import Path
from config.settings import LOGGING_CONFIG

def setup_logging():
    """Setup centralized logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_file = Path(LOGGING_CONFIG['file'])
    log_file.parent.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            # File handler with rotation
            logging.handlers.RotatingFileHandler(
                LOGGING_CONFIG['file'],
                maxBytes=LOGGING_CONFIG['max_bytes'],
                backupCount=LOGGING_CONFIG['backup_count']
            ),
            # Console handler
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    loggers = [
        'scrapers',
        'analytics',
        'database',
        'dashboard'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, LOGGING_CONFIG['level']))
    
    logging.info("Logging system initialized")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name) 