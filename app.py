#%%
import whisper
import subprocess
import yt_dlp as youtube_dl
from tqdm import tqdm
import tempfile
import os
import io

# Function to get audio URL using yt-dlp
def get_audio_url(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'extractaudio': True,
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            audio_url = info_dict['url']
            print("Audio Received")
        return audio_url
    except youtube_dl.utils.DownloadError as e:
        print(f"Error downloading audio: {e}")
        return None

# Function to transcribe audio using Whisper
def transcribe_audio(audio_url, model):
    if audio_url is None:
        return None
    
    print("ffmpeg streaming audio to Whisper")
    # Use ffmpeg to stream audio directly to Whisper
    ffmpeg_command = [
        'ffmpeg', '-i', audio_url, '-f', 'wav', '-ar', '16000', '-ac', '1', 'pipe:1'
    ]
    try:
        # Run ffmpeg and pipe the output to Whisper
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_stream = io.BytesIO(ffmpeg_process.stdout.read())
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running ffmpeg command: {e}")
        return None
    print("Transcribed")
    # Transcribe audio
    with tqdm(total=100, desc="Transcribing") as pbar:
        result = model.transcribe(audio_stream, progress_callback=lambda p: pbar.update(p * 100))

    return result['text']

# Main function
def main(youtube_url, output_txt_path):
    # Load Whisper model
    model = whisper.load_model("base")

    # Get audio URL
    audio_url = get_audio_url(youtube_url)

    # Transcribe audio
    transcription = transcribe_audio(audio_url, model)

    # Save transcription to a text file
    with open(output_txt_path, 'w') as f:
        f.write(transcription)

if __name__ == "__main__":
    youtube_url = ""
    output_txt_path = "transcripts/transcription1.txt"
    main(youtube_url, output_txt_path)

#%%