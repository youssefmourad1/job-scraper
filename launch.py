#!/usr/bin/env python3
"""
Launch script for Job Market Analyzer
"""
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import streamlit
        import pandas
        import plotly
        import aiohttp
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        return False

def install_requirements():
    """Install requirements"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def launch_app():
    """Launch the Streamlit app"""
    print("🚀 Launching Job Market Analyzer...")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "main_app.py", "--server.port=8501"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")

def main():
    """Main launch function"""
    print("🔧 Job Market Analyzer - Launch Script")
    print("=" * 50)
    
    # Check if requirements are met
    if not check_requirements():
        print("\n📦 Installing missing dependencies...")
        if not install_requirements():
            print("❌ Failed to install dependencies. Please run manually:")
            print("   pip install -r requirements.txt")
            return
    
    # Launch the application
    launch_app()

if __name__ == "__main__":
    main() 