from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from pydub import AudioSegment
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import librosa
from sklearn.preprocessing import LabelEncoder
import scipy

app = Flask(__name__)

final = pd.read_pickle("extracted_df.pkl")
y = np.array(final["name"].tolist())
le = LabelEncoder()
le.fit_transform(y)
Model1_ANN = load_model("Model1.h5")

# Define a dictionary to map bird species names to their corresponding image paths
bird_images = {
    'Song Sparrow': 'static/DATASET/photos/song sparrow.jpg',
    'Northern Mockingbird': 'static/DATASET/photos/northern mocking bird.jpg',
    'Northern Cardinal': 'static/DATASET/photos/northern cardinal.jpg',
    'American Robin': 'static/DATASET/photos/american robin.jpg',
    'Bewick\'s Wren': 'static/DATASET/photos/bewicks wren.jpg',
    # Add more bird species if needed
}

def extract_feature(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    feature_scaled = np.mean(feature.T, axis=0)
    return np.array([feature_scaled])

def identify_bird_calls(audio_path):
    # Load the audio file
    audio_data, sample_rate = librosa.load(audio_path, sr=None)

    # Compute the short-time Fourier transform (STFT)
    stft = np.abs(librosa.stft(audio_data))

    # Calculate the total energy in each frame
    frame_energy = np.sum(stft, axis=0)

    # Smooth the frame energy using a rolling mean
    smoothed_energy = np.convolve(frame_energy, np.ones(100)/100, mode='same')

    # Apply median filtering to remove noise
    smoothed_energy = scipy.signal.medfilt(smoothed_energy, kernel_size=5)

    # Find peaks in the smoothed energy
    peaks, _ = scipy.signal.find_peaks(smoothed_energy, height=np.mean(smoothed_energy) + np.std(smoothed_energy))

    # Identify segments around peaks
    segments = []
    for peak in peaks:
        segment_start = max(0, peak - int(sample_rate * 1.5))  # 1.5 seconds before the peak
        segment_end = min(len(audio_data), peak + int(sample_rate * 1.5))  # 1.5 seconds after the peak
        segments.append((segment_start, segment_end))

    # Select the segment with the most energy
    selected_segment = max(segments, key=lambda s: np.sum(frame_energy[s[0]:s[1]]))

    # Add extra padding to the start and end of the segment
    padding = int(sample_rate * 1.5)  # 1.5 seconds padding
    start_time = max(0, librosa.samples_to_time(selected_segment[0] - padding, sr=sample_rate))
    end_time = min(librosa.samples_to_time(selected_segment[1] + padding, sr=sample_rate),
                   librosa.get_duration(y=audio_data))

    return start_time, end_time

def ANN_print_prediction(audio_path):
    # Identify the bird call segment
    start_time, end_time = identify_bird_calls(audio_path)
    # Load the audio segment containing the bird call
    audio = AudioSegment.from_file(audio_path)
    # Trim the audio clip to 3 seconds containing the essential part of the bird sound
    start_time = max(0, start_time - 1)  # Start 1 second before the bird call
    end_time = min(end_time + 2, audio.duration_seconds)  # End 2 seconds after the bird call
    trimmed_audio = audio[start_time * 1000:end_time * 1000]
    # Export the trimmed segment to a temporary WAV file
    temp_wav_file_path = "static/tests/bird_call_segment.wav"
    trimmed_audio.export(temp_wav_file_path, format="wav")
    # Perform prediction on the trimmed WAV file
    prediction_feature = extract_feature(temp_wav_file_path)
    predicted_vector = np.argmax(Model1_ANN.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    image_path = bird_images.get(predicted_class[0], 'static/DATASET/photos/default_image.jpg')
    return predicted_class[0], image_path

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/login")
def login():
    return render_template('login.html')

@app.route("/index", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        # Check if the request contains the audio file
        if 'audio_file' not in request.files:
            return "No audio file found", 400
        
        # Get the audio file from the request
        audio_file = request.files['audio_file']
        
        # Check if the file is empty
        if audio_file.filename == '':
            return "No audio file selected", 400
        
        # Check if the file format is supported
        if audio_file.filename.split('.')[-1] not in ['mp3', 'wav', 'ogg', 'flac', 'mpeg', 'mp4', 'weba']:
            return "Unsupported file format", 400

        # Convert the audio to WAV format
        try:
            audio = AudioSegment.from_file(audio_file)
            wav_file_path = f"static/tests/{secure_filename(audio_file.filename)}.wav"
            audio.export(wav_file_path, format="wav")
        except Exception as e:
            return f"Error converting audio: {str(e)}", 500
        
        # Perform prediction on the trimmed WAV file
        predict_result, image_path = ANN_print_prediction(wav_file_path)
        
        # Return the prediction result
        return render_template("prediction.html", prediction=predict_result, audio_path=wav_file_path, image_path=image_path)

@app.route("/predict", methods=['POST'])
def predict():
    if 'audio_blob' not in request.files:
        return jsonify({'error': 'No audio data found'}), 400
    
    audio_blob = request.files['audio_blob']
    
    # Process and predict using the audio data
    # Implement your prediction logic here
    
    return jsonify({'prediction': 'Your prediction result'}), 200

@app.route("/chart")
def chart():
    return render_template('chart.html')

if __name__ == '__main__':
    app.run(debug=True)
