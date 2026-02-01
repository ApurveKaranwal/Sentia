import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import yt_dlp
from datetime import datetime
import tempfile
import argparse
from typing import Tuple, Optional
import warnings
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
import json
warnings.filterwarnings('ignore')

# Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:5500", "http://127.0.0.1:5500"])

# Create directories
BASE_DIR = Path("video_analysis")
BASE_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = BASE_DIR / "processed_videos"
PROCESSED_DIR.mkdir(exist_ok=True)
CSV_FILE = BASE_DIR / "analysis_results.csv"

class VideoAnalyzer:
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.gender_labels = ['Male', 'Female']
        self.age_ranges = ['18-25', '26-35', '36-45', '46-60', '60+']
        self.actions = ['speaking', 'walking', 'sitting', 'standing', 'gesturing']
        self.attributes = ['glasses', 'hat', 'beard', 'mustache', 'jewelry']
        
        # Load Haar cascades
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
    def download_youtube_clip(self, url: str, output_path: str) -> str:
        """Download YouTube video using yt-dlp (first 30s if ffmpeg available)"""
        import shutil
        has_ffmpeg = shutil.which('ffmpeg') is not None
        
        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True,
        }
        
        if has_ffmpeg:
            print("ffmpeg found, downloading first 30 seconds only...")
            ydl_opts['download_sections'] = '*0-30'
        else:
            print("ffmpeg NOT found, downloading full video (low quality forced to save bandwidth)...")
            # Force lower quality to avoid massive downloads
            ydl_opts['format'] = 'worst[ext=mp4]/worst'
        
        try:
            # ensure output path doesn't exist or is empty so yt-dlp writes to it
            if os.path.exists(output_path):
                os.remove(output_path)
                
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                 raise Exception("Download failed: File not created or empty")
                 
            return output_path
        except Exception as e:
            raise Exception(f"Failed to download YouTube video: {str(e)}")
    
    def load_video(self, video_path: str) -> Optional[cv2.VideoCapture]:
        """Load video and ensure it's only 30 seconds max"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_duration = 30  # 30 seconds
        
        # Limit to 30 seconds
        max_frames = int(fps * max_duration)
        if total_frames > max_frames:
            cap.set(cv2.CAP_PROP_FRAME_COUNT, max_frames)
        
        return cap
    
    def detect_face_emotion_gender(self, frame: np.ndarray) -> Tuple[bool, str, str, str, float]:
        """
        Enhanced face detection with emotion, gender, and age estimation
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return False, 'Neutral', 'Unknown', 'Unknown', 0.0
        
        # Use largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        
        # Enhanced emotion detection
        emotion, emotion_conf = self._analyze_emotion(face_roi, face_color)
        
        # Enhanced gender detection
        gender, gender_conf = self._analyze_gender(face_roi, face_color)
        
        # Age estimation
        age_range = self._estimate_age(face_roi)
        
        # Combine confidences
        overall_conf = (emotion_conf + gender_conf) / 2
        
    def search_and_download_videos(self, query: str, max_results: int = 5) -> list:
        """Automated scraping: Search YouTube and download videos"""
        try:
            from youtube_search import YoutubeSearch
            import requests
            
            # Search YouTube
            results = YoutubeSearch(query, max_results=max_results).to_dict()
            
            downloaded_videos = []
            for result in results:
                video_id = result['id']
                title = result['title']
                url = f"https://www.youtube.com/watch?v={video_id}"
                
                print(f"Downloading: {title}")
                
                try:
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                        video_path = self.download_youtube_clip(url, tmp.name)
                        
                        # Analyze the video
                        analysis = self.analyze_video(video_path)
                        analysis['title'] = title
                        analysis['url'] = url
                        
                        # Save to processed folder
                        dest_path = PROCESSED_DIR / f"{video_id}.mp4"
                        os.rename(video_path, dest_path)
                        
                        # Save to CSV
                        self.save_to_csv(video_id, 'scraped', analysis, self.generate_description(analysis))
                        
                        downloaded_videos.append(analysis)
                        
                except Exception as e:
                    print(f"Failed to process {title}: {e}")
                    continue
            
            return downloaded_videos
            
        except ImportError:
            print("youtube-search not installed. Install with: pip install youtube-search")
            return []
        except Exception as e:
            print(f"Scraping error: {e}")
            return []
        """Enhanced emotion analysis using multiple features"""
        # Facial feature detection
        eyes = self.eye_cascade.detectMultiScale(face_gray, 1.1, 3)
        mouths = self.mouth_cascade.detectMultiScale(face_gray, 1.7, 11)
        
        # Color analysis
        hsv = cv2.cvtColor(face_color, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1])
        
        # Texture analysis
        texture_std = np.std(face_gray)
        
        # Decision logic
        if len(mouths) > 0 and avg_saturation > 80:
            return 'Happy', 0.75
        elif texture_std > 60 and avg_hue < 20:
            return 'Angry', 0.70
        elif len(eyes) >= 2 and np.mean(face_gray) < 100:
            return 'Sad', 0.65
        elif texture_std > 40:
            return 'Surprise', 0.60
        else:
            return 'Neutral', 0.55
    
    def _analyze_gender(self, face_gray: np.ndarray, face_color: np.ndarray) -> Tuple[str, float]:
        """Enhanced gender analysis"""
        # Jawline analysis (simplified)
        height, width = face_gray.shape
        jaw_region = face_gray[int(height*0.7):height, int(width*0.2):int(width*0.8)]
        jaw_std = np.std(jaw_region)
        
        # Skin tone analysis
        avg_brightness = np.mean(face_color)
        
        # Eye region analysis
        eye_region = face_gray[int(height*0.2):int(height*0.5), :]
        eye_std = np.std(eye_region)
        
        # Decision logic (simplified heuristics)
        if jaw_std > 25 and avg_brightness > 120:
            return 'Male', 0.75
        elif eye_std < 30 and avg_brightness < 110:
            return 'Female', 0.70
        else:
            return 'Unknown', 0.50
    
    def _estimate_age(self, face_gray: np.ndarray) -> str:
        """Simple age estimation based on texture and features"""
        height, width = face_gray.shape
        
        # Texture analysis
        texture_std = np.std(face_gray)
        
        # Wrinkle estimation (simplified)
        edges = cv2.Canny(face_gray, 100, 200)
        edge_density = np.sum(edges > 0) / (height * width)
        
        # Skin smoothness
        smoothness = 1 - (texture_std / 100)
        
        # Decision logic
        if edge_density > 0.15 or texture_std > 70:
            return '46-60'
        elif texture_std > 50:
            return '36-45'
        elif smoothness > 0.7:
            return '18-25'
        else:
            return '26-35'
    def analyze_video(self, video_path: str) -> dict:
        """Main analysis pipeline with enhanced features"""
        cap = self.load_video(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = {
            'face_detected': False,
            'dominant_emotion': 'Neutral',
            'gender': 'Unknown',
            'age_range': 'Unknown',
            'confidence': 0.0,
            'actions': [],
            'attributes': [],
            'language': 'Unknown',
            'region': 'Unknown',  # Geographic region estimation
            'ethical_score': 0.0,  # Ethical compliance score
            'frame_count': total_frames
        }
        
        print("Analyzing frames...")
        frames_analyzed = 0
        motion_history = []
        
        while cap.isOpened() and frame_count < min(total_frames, 90):  # Max 3 seconds analysis
            ret, frame = cap.read()
            if not ret:
                break
                
            # Resize for faster processing
            small_frame = cv2.resize(frame, (320, 240))
            
            # Face analysis
            face_found, emotion, gender, age, conf = self.detect_face_emotion_gender(small_frame)
            
            if face_found and conf > results['confidence']:
                results['face_detected'] = True
                results['dominant_emotion'] = emotion
                results['gender'] = gender
                results['age_range'] = age
                results['confidence'] = conf
                
                # Attribute detection
                results['attributes'] = self._detect_attributes(small_frame)
            
            # Action detection (motion analysis)
            if frames_analyzed > 0:
                motion = self._detect_motion(small_frame, prev_frame)
                motion_history.append(motion)
            
            prev_frame = small_frame.copy()
            frames_analyzed += 1
            frame_count += 3  # Sample every 3rd frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        
        # Determine dominant actions
        if motion_history:
            results['actions'] = self._analyze_motion_history(motion_history)
        
        # Language detection (placeholder - would need audio analysis)
        results['language'] = self._detect_language(video_path)
        
        # Region estimation (based on language and visual cues)
        results['region'] = self._estimate_region(results['language'], results['attributes'])
        
        # Ethical compliance scoring
        results['ethical_score'] = self._calculate_ethical_score(results)
        
        cap.release()
        return results
    
    def _detect_motion(self, current_frame: np.ndarray, prev_frame: np.ndarray) -> float:
        """Calculate motion between frames"""
        if prev_frame is None:
            return 0.0
        
        # Convert to grayscale if needed
        if len(current_frame.shape) == 3:
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = current_frame
            prev_gray = prev_frame
        
        # Calculate optical flow or simple frame difference
        frame_diff = cv2.absdiff(current_gray, prev_gray)
        motion_score = np.mean(frame_diff)
        
        return motion_score
    
    def _analyze_motion_history(self, motion_history: list) -> list:
        """Analyze motion patterns to detect actions"""
        avg_motion = np.mean(motion_history)
        max_motion = np.max(motion_history)
        
        actions = []
        if max_motion > 30:
            actions.append('walking')
        elif avg_motion > 15:
            actions.append('gesturing')
        else:
            actions.append('sitting')
        
        # Add speaking if face detected (simplified assumption)
        actions.append('speaking')
        
        return list(set(actions))  # Remove duplicates
    
    def _detect_attributes(self, frame: np.ndarray) -> list:
        """Detect visible attributes like glasses, hat, etc."""
        attributes = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple heuristics for attribute detection
        height, width = gray.shape
        
        # Glasses detection (simplified)
        eye_region = gray[int(height*0.2):int(height*0.5), int(width*0.2):int(width*0.8)]
        if eye_region.size > 0:
            eye_std = np.std(eye_region)
            if eye_std > 40:
                attributes.append('glasses')
        
        # Hat detection (dark region at top)
        top_region = gray[:int(height*0.3), :]
        if np.mean(top_region) < 80:
            attributes.append('hat')
        
        # Beard detection (texture in lower face region)
        beard_region = gray[int(height*0.6):, int(width*0.3):int(width*0.7)]
        if beard_region.size > 0 and np.std(beard_region) > 35:
            attributes.append('beard')
        
            return attributes
    
    def _estimate_region(self, language: str, attributes: list) -> str:
        """Estimate geographic region based on language and visual cues"""
        if language == "Hindi":
            return "India (North/Central)"
        elif language == "English":
            # Check for Indian attributes
            if any(attr in ['bindi', 'sari', 'turban'] for attr in attributes):
                return "India"
            else:
                return "Global/Unknown"
        else:
            return "Unknown"
    
    def _calculate_ethical_score(self, results: dict) -> float:
        """Calculate ethical compliance score (0-1)"""
        score = 1.0
        
        # Deduct for potential privacy issues
        if not results['face_detected']:
            score -= 0.1  # No faces detected is actually better for privacy
        
        # Deduct for low confidence (uncertain predictions)
        if results['confidence'] < 0.5:
            score -= 0.2
        
        # Deduct for unknown demographics (better to be uncertain than wrong)
        if results['gender'] == 'Unknown':
            score -= 0.1
        if results['age_range'] == 'Unknown':
            score -= 0.1
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, score))
    
    def _detect_language(self, video_path: str) -> str:
        """Language detection using speech recognition"""
        try:
            import speech_recognition as sr
            try:
                from moviepy.editor import VideoFileClip
            except ImportError:
                try:
                    from moviepy import VideoFileClip
                except ImportError:
                    print("moviepy not found or incompatible version")
                    return "Language detection unavailable (missing moviepy)"
            
            # Extract audio from video
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            
            if audio_clip is None:
                return "No audio detected"
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
                audio_clip.write_audiofile(tmp_audio.name, verbose=False, logger=None)
                audio_path = tmp_audio.name
            
            # Speech recognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                
                # Try Hindi first
                try:
                    text = recognizer.recognize_google(audio_data, language='hi-IN')
                    if text:
                        return "Hindi"
                except:
                    pass
                
                # Try English
                try:
                    text = recognizer.recognize_google(audio_data, language='en-IN')
                    if text:
                        return "English"
                except:
                    pass
                
                # Try general English
                try:
                    text = recognizer.recognize_google(audio_data)
                    if text:
                        return "English"
                except:
                    pass
            
            # Clean up
            os.unlink(audio_path)
            video_clip.close()
            
            return "Language not detected"
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return "Language detection unavailable"
    def generate_description(self, results: dict) -> str:
        """Generate comprehensive, safe, probabilistic description"""
        if not results['face_detected']:
            return "The video shows general content without clearly detectable faces. No behavioral insights available."
        
        descriptions = []
        
        # Basic person description
        gender_desc = f"appears to be {results['gender'].lower()}" if results['gender'] != 'Unknown' else "appears to be a person"
        age_desc = f" in the {results['age_range']} age range" if results['age_range'] != 'Unknown' else ""
        descriptions.append(f"The video likely shows {gender_desc}{age_desc}.")
        
        # Emotion description
        emotion_templates = {
            'happy': "The person appears happy and engaged.",
            'sad': "The person seems to have a sad or melancholic expression.",
            'angry': "The person appears angry or frustrated.",
            'neutral': "The person maintains a neutral expression.",
            'surprise': "The person appears surprised or astonished.",
            'fear': "The person seems fearful or anxious.",
            'disgust': "The person appears disgusted or displeased."
        }
        
        emotion_desc = emotion_templates.get(results['dominant_emotion'].lower(), "The person shows an unclear emotional state.")
        descriptions.append(emotion_desc)
        
        # Actions description
        if results['actions']:
            action_str = ", ".join(results['actions'])
            descriptions.append(f"The person appears to be {action_str}.")
        
        # Attributes description
        if results['attributes']:
            attr_str = ", ".join(results['attributes'])
            descriptions.append(f"Visible attributes include: {attr_str}.")
        
        # Language description
        if results['language'] != 'Unknown':
            descriptions.append(f"The spoken language appears to be {results['language']}.")
        
        # Region description
        if results['region'] != 'Unknown':
            descriptions.append(f"Estimated geographic region: {results['region']}.")
        
        # Confidence and ethical disclaimer
        confidence_pct = int(results['confidence'] * 100)
        ethical_pct = int(results['ethical_score'] * 100)
        descriptions.append(f"\nConfidence level: Approximately {confidence_pct}%. Ethical compliance: {ethical_pct}%. ")
        descriptions.append("⚠️ IMPORTANT: These are AI-generated behavioral insights based on computer vision analysis. ")
        descriptions.append("Predictions are probabilistic and may not be accurate. No personal identification is performed. ")
        descriptions.append("This analysis is for research purposes only and should not be used for discriminatory practices.")
        
        return " ".join(descriptions)

def get_video_id(video_path: str) -> str:
    """Generate unique video ID from filename or timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = Path(video_path).stem
    return f"{filename}_{timestamp}"

def save_to_csv(video_id: str, input_method: str, results: dict, description: str):
    """Save comprehensive analysis results to CSV"""
    data = {
        'Video_ID': video_id,
        'Input_Method': input_method,
        'Emotion': results['dominant_emotion'],
        'Gender': results['gender'],
        'Age_Range': results['age_range'],
        'Confidence': f"{results['confidence']:.2f}",
        'Actions': ', '.join(results['actions']) if results['actions'] else 'None',
        'Attributes': ', '.join(results['attributes']) if results['attributes'] else 'None',
        'Language': results['language'],
        'Region': results['region'],
        'Ethical_Score': f"{results['ethical_score']:.2f}",
        'Face_Detected': results['face_detected'],
        'Frame_Count': results['frame_count'],
        'Description': description,
        'Timestamp': datetime.now().isoformat()
    }
    
    # Create DataFrame and append to CSV
    df_new = pd.DataFrame([data])
    
    if CSV_FILE.exists():
        df_existing = pd.read_csv(CSV_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(CSV_FILE, index=False)
    print(f"✅ Results saved to {CSV_FILE}")

@app.route('/', methods=['GET'])
def status():
    """Server status endpoint"""
    return jsonify({
        'status': 'running',
        'version': '2.0',
        'features': [
            'emotion_detection', 'gender_estimation', 'age_estimation',
            'action_recognition', 'attribute_detection', 'language_detection',
            'region_estimation', 'ethical_scoring', 'automated_scraping',
            'advanced_filtering'
        ],
        'endpoints': [
            'POST /analyze', 'GET /history', 'GET /download_csv',
            'GET /filter', 'POST /scrape', 'GET /'
        ]
    })

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """Web endpoint for video analysis"""
    try:
        analyzer = VideoAnalyzer()
        
        if 'video_file' in request.files:
            # Handle uploaded file
            video_file = request.files['video_file']
            input_method = request.form.get('input_method', 'manual')
            
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                video_file.save(tmp.name)
                video_path = tmp.name
        elif 'youtube_url' in request.form:
            # Handle YouTube URL
            youtube_url = request.form['youtube_url']
            input_method = request.form.get('input_method', 'youtube')
            
            # Create a localized temp path but don't hold the file open
            fd, temp_path = tempfile.mkstemp(suffix='.mp4')
            os.close(fd) # Close it immediately to let other processes use it
            
            video_path = analyzer.download_youtube_clip(youtube_url, temp_path)
        else:
            return jsonify({'error': 'No video file or YouTube URL provided'}), 400
        
        # Analyze video
        results = analyzer.analyze_video(video_path)
        description = analyzer.generate_description(results)
        video_id = get_video_id(video_path)
        
        # Move to processed folder
        dest_path = PROCESSED_DIR / f"{video_id}.mp4"
        if input_method == "youtube":
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(video_path, dest_path)
        else:
            import shutil
            shutil.copy2(video_path, dest_path)
        
        # Save to CSV
        save_to_csv(video_id, input_method, results, description)
        
        # Prepare response
        response_data = {
            'video_id': video_id,
            'emotion': results['dominant_emotion'],
            'gender': results['gender'],
            'age_range': results['age_range'],
            'confidence': results['confidence'],
            'actions': results['actions'],
            'attributes': results['attributes'],
            'language': results['language'],
            'face_detected': results['face_detected'],
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        
        # Clean up temp file if it still exists
        if os.path.exists(video_path) and video_path != str(dest_path):
            os.unlink(video_path)
        
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    try:
        if not CSV_FILE.exists():
            return jsonify([])
        
        df = pd.read_csv(CSV_FILE)
        # Convert to list of dicts, get last 50 entries
        history = df.tail(50).to_dict('records')
        return jsonify(history)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_csv', methods=['GET'])
def download_csv():
    """Download the CSV file"""
    try:
        if not CSV_FILE.exists():
            return jsonify({'error': 'No data available'}), 404
        
        return send_file(CSV_FILE, as_attachment=True, download_name='analysis_results.csv')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/filter', methods=['GET'])
def filter_results():
    """Advanced filtering endpoint"""
    try:
        emotion = request.args.get('emotion')
        gender = request.args.get('gender')
        age_range = request.args.get('age_range')
        region = request.args.get('region')
        
        if not CSV_FILE.exists():
            return jsonify([])
        
        df = pd.read_csv(CSV_FILE)
        
        # Apply filters
        if emotion:
            df = df[df['Emotion'].str.lower() == emotion.lower()]
        if gender:
            df = df[df['Gender'].str.lower() == gender.lower()]
        if age_range:
            df = df[df['Age_Range'] == age_range]
        if region:
            df = df[df['Region'].str.contains(region, case=False, na=False)]
        
        results = df.to_dict('records')
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scrape', methods=['POST'])
def scrape_videos():
    """Automated scraping endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', 'Indian speakers')
        max_results = min(int(data.get('max_results', 3)), 10)  # Limit to 10 max
        
        analyzer = VideoAnalyzer()
        results = analyzer.search_and_download_videos(query, max_results)
        
        return jsonify({
            'message': f'Successfully scraped and analyzed {len(results)} videos',
            'results': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    parser = argparse.ArgumentParser(description="Video Behavior Analysis System")
    parser.add_argument('--youtube', type=str, help="YouTube URL")
    parser.add_argument('--file', type=str, help="Local video file path")
    parser.add_argument('--web', action='store_true', help="Run as web server")
    parser.add_argument('--port', type=int, default=5000, help="Port for web server")
    args = parser.parse_args()
    
    if args.web:
        # Run as web server
        print("🚀 Starting Video Behavior Analysis Web Server...")
        print(f"📡 Server will run on http://localhost:{args.port}")
        print("📁 Frontend should be served separately (e.g., using a web server)")
        app.run(host='0.0.0.0', port=args.port, debug=True)
        return
    
    # CLI mode
    analyzer = VideoAnalyzer()
    
    if args.youtube:
        print("📥 Downloading YouTube video (first 30 seconds)...")
        input_method = "YouTube"
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            video_path = analyzer.download_youtube_clip(args.youtube, temp_path)
        except Exception as e:
            print(f"❌ Error: {e}")
            return
    
    elif args.file and os.path.exists(args.file):
        print("📁 Loading local video file...")
        input_method = "Manual"
        video_path = args.file
    else:
        print("❌ Please provide either --youtube URL or --file path")
        print("💡 Or use --web to start the web server")
        return
    
    # Process video
    try:
        print("🔍 Analyzing video...")
        results = analyzer.analyze_video(video_path)
        description = analyzer.generate_description(results)
        video_id = get_video_id(video_path)
        
        # Move/copy to processed folder
        dest_path = PROCESSED_DIR / f"{video_id}.mp4"
        if input_method == "YouTube":
            os.rename(video_path, dest_path)
        else:
            import shutil
            shutil.copy2(video_path, dest_path)
        
        # Display results
        print("\n" + "="*60)
        print("🎥 VIDEO ANALYSIS RESULTS")
        print("="*60)
        print(f"Face detected: {'✅ Yes' if results['face_detected'] else '❌ No'}")
        print(f"Dominant emotion: {results['dominant_emotion']}")
        print(f"Estimated gender: {results['gender']}")
        print(f"Age range: {results['age_range']}")
        print(f"Actions: {', '.join(results['actions']) if results['actions'] else 'None detected'}")
        print(f"Attributes: {', '.join(results['attributes']) if results['attributes'] else 'None detected'}")
        print(f"Language: {results['language']}")
        print(f"Confidence: {results['confidence']:.2f}")
        print(f"\n📝 Description: {description}")
        print(f"💾 Stored as: {dest_path.name}")
        
        # Save to CSV
        save_to_csv(video_id, input_method, results, description)
        
        print(f"\n✅ Analysis complete! Check {CSV_FILE} for all results.")
        print("\n⚠️  Note: These are approximate predictions based on computer vision models.")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get analysis statistics"""
    try:
        if not CSV_FILE.exists():
            return jsonify({
                'total': 0,
                'avg_confidence': 0,
                'top_emotion': 'None',
                'analysis_rate': '0/day'
            })

        df = pd.read_csv(CSV_FILE)

        # Calculate stats
        total = len(df)
        avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 0
        top_emotion = df['emotion'].mode().iloc[0] if len(df) > 0 and 'emotion' in df.columns else 'None'

        # Calculate analysis rate (analyses per day)
        if len(df) > 0 and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            days_active = (df['timestamp'].max() - df['timestamp'].min()).days
            days_active = max(days_active, 1)  # Avoid division by zero
            analysis_rate = f"{total / days_active:.1f}/day"
        else:
            analysis_rate = "0/day"

        return jsonify({
            'total': int(total),
            'avg_confidence': float(avg_confidence),
            'top_emotion': top_emotion,
            'analysis_rate': analysis_rate
        })

    except Exception as e:
        print(f"Stats error: {e}")
        return jsonify({'error': 'Failed to get statistics'}), 500

@app.route('/export_selected', methods=['POST'])
def export_selected():
    """Export selected analysis results to CSV"""
    try:
        data = request.get_json()
        selected_ids = data.get('ids', [])

        if not CSV_FILE.exists():
            return jsonify({'error': 'No data available'}), 404

        df = pd.read_csv(CSV_FILE)
        df['id'] = df.index.astype(str)  # Add ID column for matching

        # Filter selected rows
        selected_df = df[df['id'].isin(selected_ids)]

        if selected_df.empty:
            return jsonify({'error': 'No matching records found'}), 404

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            selected_df.drop('id', axis=1).to_csv(tmp.name, index=False)
            tmp_path = tmp.name

        return send_file(tmp_path, as_attachment=True, download_name='selected_analyses.csv')

    except Exception as e:
        print(f"Export error: {e}")
        return jsonify({'error': 'Export failed'}), 500

@app.route('/delete_selected', methods=['POST'])
def delete_selected():
    """Delete selected analysis results"""
    try:
        data = request.get_json()
        selected_ids = data.get('ids', [])

        if not CSV_FILE.exists():
            return jsonify({'error': 'No data available'}), 404

        df = pd.read_csv(CSV_FILE)
        df['id'] = df.index.astype(str)

        # Remove selected rows
        filtered_df = df[~df['id'].isin(selected_ids)]

        # Save back to CSV
        filtered_df.drop('id', axis=1).to_csv(CSV_FILE, index=False)

        return jsonify({'message': f'Successfully deleted {len(selected_ids)} records'})

    except Exception as e:
        print(f"Delete error: {e}")
        return jsonify({'error': 'Delete failed'}), 500

@app.route('/download_results', methods=['POST'])
def download_results():
    """Download current analysis results as JSON"""
    try:
        data = request.get_json()

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(data, tmp, indent=2)
            tmp_path = tmp.name

        return send_file(tmp_path, as_attachment=True, download_name='analysis_results.json')

    except Exception as e:
        print(f"Download error: {e}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/save_results', methods=['POST'])
def save_results():
    """Save current analysis results to history"""
    try:
        data = request.get_json()

        # Add timestamp and save to CSV
        data['timestamp'] = datetime.now().isoformat()
        data['input_method'] = data.get('input_method', 'manual')

        # Load existing data or create new dataframe
        if CSV_FILE.exists():
            df = pd.read_csv(CSV_FILE)
        else:
            df = pd.DataFrame()

        # Append new result
        new_row = pd.DataFrame([data])
        df = pd.concat([df, new_row], ignore_index=True)

        # Save to CSV
        df.to_csv(CSV_FILE, index=False)

        return jsonify({'message': 'Results saved successfully'})

    except Exception as e:
        print(f"Save error: {e}")
        return jsonify({'error': 'Save failed'}), 500
