#!/usr/bin/env python3
"""
Simple server script to run the Video Analysis Flask app
"""
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from main import app

if __name__ == '__main__':
    print("🚀 Starting Video Behavior Analysis Web Server...")
    print("📡 Server will run on http://localhost:5000")
    print("📁 Frontend should be served separately")
    app.run(host='0.0.0.0', port=5000, debug=True)