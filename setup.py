#!/usr/bin/env python3
"""
Setup and initialization script for Resume Screening System
Run this once to set up the project
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_project():
    """Main setup function"""
    print("🚀 Resume Screening System - Setup")
    print("=" * 50)
    
    # Check Python version
    print("\n📋 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]}")
    
    # Check if venv exists
    print("\n🔧 Setting up virtual environment...")
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created")
    else:
        print("✅ Virtual environment already exists")
    
    # Activate venv and install requirements
    print("\n📦 Installing dependencies...")
    
    # Different activation command for Windows vs Unix
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    # Install requirements
    try:
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download NLTK data
    print("\n📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ Warning: Could not download NLTK data: {e}")
    
    # Create directories
    print("\n📁 Creating project directories...")
    directories = ["resumes", "job_descriptions", "screenshots"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")
    
    # Copy .env.example to .env if not exists
    print("\n🔐 Setting up environment configuration...")
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("✅ .env file created (configure OPENAI_API_KEY if needed)")
    elif not env_file.exists():
        print("⚠️ .env file not found")
    else:
        print("✅ .env file already exists")
    
    # Final instructions
    print("\n" + "=" * 50)
    print("✅ Setup complete!")
    print("\n🚀 To start the application:")
    
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("   streamlit run app.py")
    print("\n📖 See README.md for detailed documentation")
    print("=" * 50)


if __name__ == "__main__":
    try:
        setup_project()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
