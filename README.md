# Intelligent Video Analysis System

An AI-powered system for analyzing behavioral insights from public YouTube videos, specifically designed for Indian speakers using Hindi and English. The system processes short video clips (up to 30 seconds) and extracts comprehensive behavioral data while maintaining strict ethical standards.

## Features

### Core Analysis Capabilities
- **Emotion Detection**: Identifies dominant emotions (Happy, Sad, Angry, Neutral, Surprised, etc.)
- **Demographic Analysis**: Estimates gender and age range
- **Action Recognition**: Detects observable actions (speaking, walking, sitting, gesturing)
- **Attribute Detection**: Identifies visible attributes (glasses, hat, beard, jewelry)
- **Language Detection**: Recognizes Hindi/English speech patterns

### Advanced Features
- **Post-processing Filters**: Filter results by emotion, gender, age range, and region
- **Ethical AI Layer**: Built-in safeguards against personal identification
- **Probabilistic Reporting**: Uses safe, natural language with confidence indicators
- **Structured Data Storage**: Mandatory CSV tracking for all analyses

### Input Methods
- YouTube video URLs (automatically downloads first 30 seconds)
- Direct video file uploads (MP4, MOV, AVI formats)

## Installation

1. **Clone or download the project**
   ```bash
   cd "d:\VS Code Projects\REAL PROJECTS\Sentia"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure OpenCV data files are available**
   The system uses OpenCV's built-in Haar cascades for face detection.

## Usage

### Web Interface (Recommended)
1. **Start the backend server**
   ```bash
   python backend/main.py --web --port 5000
   ```

2. **Serve the frontend**
   Use any static file server or open `frontend/index.html` directly in a browser.
   ```bash
   # Example using Python's built-in server
   cd frontend
   python -m http.server 8000
   ```

3. **Access the application**
   - Backend API: http://localhost:5000
   - Frontend: http://localhost:8000

### Command Line Interface
```bash
# Analyze YouTube video
python backend/main.py --youtube "https://youtube.com/watch?v=VIDEO_ID"

# Analyze local video file
python backend/main.py --file "path/to/video.mp4"
```

## API Endpoints

- `POST /analyze`: Analyze a video (accepts YouTube URL or file upload)
- `GET /history`: Retrieve analysis history
- `GET /download_csv`: Download complete results as CSV
- `GET /filter`: Advanced filtering with query parameters

## Output Data Structure

Each analysis generates:
- **Video ID**: Unique identifier
- **Behavioral Insights**: Emotion, gender, age range, actions, attributes
- **Technical Data**: Confidence scores, frame count, language detection
- **Ethical Description**: Safe, probabilistic natural language summary
- **Metadata**: Timestamp, input method, processing details

## Ethical Considerations

### Privacy Protection
- No facial recognition or personal identification
- No storage of biometric data
- Analysis limited to behavioral patterns only

### Responsible AI
- Clear disclosure of AI limitations and assumptions
- Probabilistic language avoids definitive claims
- Designed for research and analytical purposes only

### Data Handling
- Videos processed locally (no external API calls for analysis)
- Results stored in structured CSV format
- No raw video data retained beyond processing

## Technical Architecture

### Backend (Python)
- **Computer Vision**: OpenCV with Haar cascades
- **Video Processing**: yt-dlp for YouTube downloads
- **Web Framework**: Flask with CORS support
- **Data Storage**: Pandas CSV management

### Frontend (Web)
- **Interface**: HTML5/CSS3/JavaScript
- **Styling**: Modern gradient design with responsive layout
- **Features**: Real-time analysis, history viewing, CSV export

### Analysis Pipeline
1. Video acquisition (YouTube download or file upload)
2. Frame extraction and preprocessing
3. Face detection using Haar cascades
4. Multi-feature analysis (emotion, gender, age, actions, attributes)
5. Result aggregation and description generation
6. CSV storage and API response

## Limitations

### Technical Constraints
- Analysis based on computer vision heuristics (not deep learning models)
- Limited to visible facial features and basic motion detection
- Age/gender estimation accuracy depends on image quality
- Language detection is placeholder (requires audio processing for production)

### Ethical Boundaries
- Designed for behavioral research, not individual identification
- Results should not be used for discriminatory purposes
- Always consider human oversight for critical applications

## Future Enhancements

- Integration with deep learning models (FER, DeepFace)
- Audio analysis for accurate language detection
- Real-time video streaming analysis
- Advanced ML-based action recognition
- Multi-language support beyond Hindi/English
- Cloud deployment options

## License

This project is intended for educational and research purposes. Please ensure compliance with YouTube's Terms of Service when analyzing public videos.

## Disclaimer

This system provides approximate behavioral insights based on computer vision analysis. Results are probabilistic and should be interpreted cautiously. The system is not intended for personal identification or any form of surveillance.