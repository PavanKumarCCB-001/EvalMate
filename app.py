from flask import Flask, request, send_from_directory, jsonify
import os
import cv2
import subprocess
import speech_recognition as sr
import google.generativeai as google_genai

app = Flask(__name__, static_url_path='', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configure Google GenAI with your API key
google_genai.configure(api_key='Your API Key')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.htm')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({"error": "No video selected"}), 400
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(save_path)
    process_response = app.test_client().post('/process', json={'filename': video.filename})
    if process_response.status_code == 200:
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
        try:
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_path) as source:
                audio_data = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_data)
                model = google_genai.GenerativeModel('gemini-1.5-flash-001')
                prompt = f"Analyze this transcription for communication skills and provide feedback: {transcription}"
                response = model.generate_content(prompt)
                feedback = response.text
                return jsonify({"message": "Video uploaded, audio extracted, transcribed, and analyzed", "filename": video.filename, "path": save_path, "transcription": transcription, "feedback": feedback}), 200
        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand audio"}), 500
        except sr.RequestError as e:
            return jsonify({"error": f"Speech recognition error: {str(e)}"}), 500
        except Exception as e:
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    return jsonify({"message": "Video uploaded, but processing failed", "filename": video.filename, "path": save_path, "error": process_response.get_json()['error']}), 500

@app.route('/process', methods=['POST'])
def process_video():
    data = request.get_json()
    if not data or 'filename' not in data:
        return jsonify({"error": "No filename provided"}), 400
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'audio.wav')
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return jsonify({"error": "Could not open video"}), 500
    command = [
        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
        audio_path, '-y'
    ]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Audio extraction failed: {str(e)}"}), 500
    finally:
        video.release()
    
    return jsonify({"message": "Audio extracted successfully", "audio_path": audio_path}), 200

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    data = request.get_json()
    if not data or 'audio_path' not in data:
        return jsonify({"error": "No audio path provided"}), 400
    audio_path = data['audio_path']
    if not os.path.exists(audio_path):
        return jsonify({"error": "Audio file not found"}), 404
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            transcription = recognizer.recognize_google(audio_data)
            return jsonify({"message": "Transcription successful", "transcription": transcription}), 200
        except sr.UnknownValueError:
            return jsonify({"error": "Could not understand audio"}), 500
        except sr.RequestError as e:
            return jsonify({"error": f"Speech recognition error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
