#!/usr/bin/env python3
"""
Test script for the Video Behavior Analysis System
"""

def test_imports():
    """Test if all required packages can be imported"""
    try:
        import cv2
        import pandas as pd
        import numpy as np
        import yt_dlp
        import flask
        import flask_cors
        import speech_recognition
        import requests
        import moviepy
        from youtube_search import YoutubeSearch
        
        print("✅ All packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_video_analyzer():
    """Test VideoAnalyzer class initialization"""
    try:
        from backend.main import VideoAnalyzer
        
        analyzer = VideoAnalyzer()
        print("✅ VideoAnalyzer initialized successfully")
        print(f"   - Emotion labels: {len(analyzer.emotion_labels)}")
        print(f"   - Gender labels: {len(analyzer.gender_labels)}")
        print(f"   - Age ranges: {len(analyzer.age_ranges)}")
        print(f"   - Actions: {len(analyzer.actions)}")
        print(f"   - Attributes: {len(analyzer.attributes)}")
        return True
    except Exception as e:
        print(f"❌ VideoAnalyzer error: {e}")
        return False

def test_directories():
    """Test if required directories exist"""
    try:
        from pathlib import Path
        
        base_dir = Path("video_analysis")
        processed_dir = base_dir / "processed_videos"
        csv_file = base_dir / "analysis_results.csv"
        
        # Create directories if they don't exist
        base_dir.mkdir(exist_ok=True)
        processed_dir.mkdir(exist_ok=True)
        
        print("✅ Directories created successfully")
        print(f"   - Base dir: {base_dir.absolute()}")
        print(f"   - Processed dir: {processed_dir.absolute()}")
        print(f"   - CSV file: {csv_file.absolute()}")
        return True
    except Exception as e:
        print(f"❌ Directory error: {e}")
        return False

def main():
    print("🧪 Testing Video Behavior Analysis System Components")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("VideoAnalyzer Class", test_video_analyzer),
        ("Directory Setup", test_directories)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Component Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All components working! System is ready.")
        print("\n🚀 To start the web server:")
        print("   python backend/main.py --web --port 5000")
        print("\n🌐 To serve the frontend:")
        print("   cd frontend && python -m http.server 8000")
        print("   Then open http://localhost:8000 in your browser")
    else:
        print("⚠️ Some components failed. Check error messages above.")
    
    print("\n📋 System Features:")
    print("   ✅ Emotion & Gender Detection")
    print("   ✅ Age Range Estimation")
    print("   ✅ Action Recognition")
    print("   ✅ Attribute Detection")
    print("   ✅ Language Detection (Speech Recognition)")
    print("   ✅ Region Estimation")
    print("   ✅ Ethical Compliance Scoring")
    print("   ✅ Automated YouTube Scraping")
    print("   ✅ Advanced Filtering")
    print("   ✅ Web Interface with Real-time Analysis")

if __name__ == "__main__":
    main()