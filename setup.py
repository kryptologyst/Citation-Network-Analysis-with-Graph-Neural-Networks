#!/usr/bin/env python3
"""Quick setup and test script for the citation network analysis project."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("✗ Python 3.8+ is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def main():
    """Main setup function."""
    print("Citation Network Analysis - Quick Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("configs").exists():
        print("✗ Please run this script from the project root directory")
        sys.exit(1)
    
    print("✓ Project structure detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("✗ Failed to install dependencies")
        sys.exit(1)
    
    # Run basic tests
    if not run_command("python -m pytest tests/test_basic.py -v", "Running basic tests"):
        print("✗ Tests failed")
        sys.exit(1)
    
    # Run example script
    if not run_command("python example.py", "Running example script"):
        print("✗ Example script failed")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Train a model: python scripts/train.py model=gcn data=cora")
    print("2. Compare models: python scripts/compare_models.py data=cora")
    print("3. Launch demo: streamlit run demo/app.py")
    print("4. Read the README.md for more information")


if __name__ == "__main__":
    main()
