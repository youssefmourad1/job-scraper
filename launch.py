#!/usr/bin/env python3
"""
Launch script for the comprehensive Job Market Analysis Platform
All-in-one Streamlit application with scraping, EDA, topic modeling, and AI reports
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'scikit-learn',
        'aiohttp', 'beautifulsoup4', 'requests', 'python-dotenv',
        'fpdf2', 'nltk'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Run: pip install -r requirements.txt")
        return False
    
    return True

def setup_environment():
    """Set up the environment for the application"""
    # Create necessary directories
    directories = ['data', 'reports', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Initialize database if it doesn't exist
    try:
        from config.database import DatabaseManager
        db_manager = DatabaseManager()
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False
    
    return True

def launch_application():
    """Launch the Streamlit application"""
    try:
        # Set Streamlit configuration
        os.environ['STREAMLIT_SERVER_PORT'] = '8501'
        os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
        
        # Launch the main application
        logger.info("🚀 Launching Job Market Analysis Platform...")
        logger.info("📊 Access the application at: http://localhost:8501")
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'main_app.py',
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--server.headless=false',
            '--browser.gatherUsageStats=false'
        ])
        
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to launch application: {e}")

def main():
    """Main launch function"""
    print("="*80)
    print("🚀 JOB MARKET ANALYSIS PLATFORM")
    print("="*80)
    print("📊 Comprehensive analysis with scraping, EDA, ML & AI")
    print("🤖 Powered by Streamlit, scikit-learn & Grok AI")
    print("="*80)
    
    # Check dependencies
    # print("\n🔍 Checking dependencies...")
    # if not check_dependencies():
    #     print("❌ Please install missing dependencies")
    #     return
    # print("✅ All dependencies found")
    
    # Setup environment
    print("\n⚙️ Setting up environment...")
    if not setup_environment():
        print("❌ Environment setup failed")
        return
    print("✅ Environment ready")
    
    # Launch application
    print("\n🚀 Launching application...")
    launch_application()

if __name__ == "__main__":
    main() 