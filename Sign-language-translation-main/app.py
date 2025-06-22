from flask import Flask, request, jsonify, send_file, render_template
import os
import time
from PIL import Image
from moviepy.editor import ImageSequenceClip, concatenate_videoclips,ImageClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import speech_recognition as sr
import pyttsx3
import base64
from pydub import AudioSegment
from flask_cors import CORS
from flask import Flask, render_template, Response
import cv2
import numpy as np
from gtts import gTTS
import os
import tensorflow as tf
import moviepy.video.fx.all as vfx

model = tf.keras.models.load_model('audio_to_gesture_model.h5')
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

word_gestures_folder = "filtered_data"
alphabet_gestures_folder = "alphabet"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/audio_to_gesture')
def audio_to_gesture():
    return render_template('audio_to_gesture.html')

def text_to_gesture_sequence(input_text):
    words = input_text.lower().split()  
    gesture_sequence = []
    for word in words:
        word_gesture_path = os.path.join(word_gestures_folder, f"{word}.gif") 
        print(word_gesture_path)
        if os.path.exists(word_gesture_path):
            gesture_sequence.append(word_gesture_path)
        else:
            for char in word:
                alphabet_gesture_path = os.path.join(alphabet_gestures_folder, f"{char}.gif")
                if os.path.exists(alphabet_gesture_path):
                    gesture_sequence.append(alphabet_gesture_path)
                else:
                    print(f"Warning: No gesture image found for letter: {char}")
    return gesture_sequence
def convert_webp_to_gif(input_dir, output_dir):
    # List all .webp files in the input directory
    webp_files = [f for f in os.listdir(input_dir) if f.endswith('.webp')]
    gif_files = []
    
    for webp_file in webp_files:
        # Open the .webp file
        with Image.open(os.path.join(input_dir, webp_file)) as img:
            # Define the output .gif file path
            gif_file = os.path.splitext(webp_file)[0] + '.gif'
            # Save as .gif
            img.save(os.path.join(output_dir, gif_file), 'GIF')
            gif_files.append(os.path.join(output_dir, gif_file))
            print(f'Converted {webp_file} to {gif_file}')
    
    return gif_files
def create_gif_from_images(gif_files, output_file, resize_to=(200, 200), frame_duration=0.1):
    convert_webp_to_gif(word_gestures_folder, word_gestures_folder)
    clips = [ImageSequenceClip([gif], fps=1) for gif in gif_files]
    video = concatenate_videoclips(clips, method="compose")
    playback_speed = 0.8  
    video = video.fx(vfx.speedx, playback_speed)
    video.write_videofile(output_file, codec="libx264")
    

def predict_gesture_class(image_path):
    image = Image.open(image_path).resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    predictions = model.predict(image_array)
    gesture_class = np.argmax(predictions, axis=1)
    return gesture_class

def generate_gesture_gif(input_text, output_gif_path):
    start_time = time.time()
    gesture_sequence = text_to_gesture_sequence(input_text)
    print(gesture_sequence)
    create_gif_from_images(gesture_sequence, output_gif_path, frame_duration=-1.0)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Time taken to generate GIF: {execution_time:.2f} seconds")

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
from datetime import datetime

@app.route('/speech', methods=['POST'])
def transcribe_speech_from_mic():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)

        with open("transcription.txt", "w") as file:
            file.write(text)
        
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand audio"
    except sr.RequestError:
        return "Could not request results from Google Speech Recognition service"
    
    
    return render_template('audio_to_gesture.html', video_url="")

@app.route('/process_audio', methods=['POST'])
def process_audio():
    with open("transcription.txt", "r") as file:
        text = file.read().strip()

    if 'audio_file' in request.files:
        audio_file = request.files['audio_file']
        if audio_file:
            audio_filename = "input_audio.mp3"
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)
            wav_path = os.path.join(app.config['UPLOAD_FOLDER'], "input_audio.wav")

            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            audio_file.save(audio_path)
            convert_mp3_to_wav(audio_path, wav_path)

    else:
        return jsonify({"error": "No valid audio file provided"}), 400
    
    if text:
        input_text = text
    
    output_gif_path = os.path.join("static", "final_output_gesture.mp4")
    generate_gesture_gif(input_text, output_gif_path)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    video_url = f"/static/final_output_gesture.mp4?t={timestamp}"

    return render_template('audio_to_gesture.html', video_url=video_url)


if __name__ == "__main__":
    app.run(debug=True)
