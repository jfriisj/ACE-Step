#!/usr/bin/env python3
"""
ACE-Step Audio Diagnostics - Main Entry Point
Provides easy access to audio file diagnostics for ACE-Step generated files
"""

# This is a simple wrapper to make diagnostics more discoverable
import sys
import os

# Import and run the main diagnostic runner
if __name__ == '__main__':
    # Add the current directory to Python path to ensure imports work
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Import and run the diagnostic runner
    from diagnostic_runner import main
    sys.exit(main())