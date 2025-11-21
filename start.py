"""
Quick Start Script for Tone Converter Application
Run this script to start the application quickly
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import flask_cors
        print("✓ Required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("   Run: pip install -r requirements.txt")
        return False

def start_application():
    """Start the Flask application"""
    print("\n" + "="*60)
    print("🚀 Starting Tone Converter Application")
    print("="*60)
    
    if not check_python_version():
        return
    
    if not check_dependencies():
        install = input("\nDo you want to install dependencies now? (y/n): ")
        if install.lower() == 'y':
            print("\nInstalling dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        else:
            return
    
    print("\n📍 Server will start at: http://localhost:5000")
    print("📊 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    # Start Flask app
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    start_application()
