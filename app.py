import whisper
import yt_dlp as youtube_dl
import os

# Function to extract audio stream from YouTube
def extract_audio_stream(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': 'temp_audio.%(ext)s',
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            audio_path = ydl.prepare_filename(info_dict)
            # Ensure the file is in wav format
            if not audio_path.endswith('.wav'):
                audio_path = audio_path.rsplit('.', 1)[0] + '.wav'
            return audio_path
    except youtube_dl.utils.DownloadError as e:
        print(f"Error extracting audio: {e}")
        return None

# Function to transcribe an audio stream using Whisper
def transcribe_stream(audio_file_path, model):
    audio_data = model.transcribe(audio_file_path, language="english")
    result = audio_data["text"]
    return result

# Main function
def main(youtube_url, output_txt_path):
    # Load Whisper model
    model = whisper.load_model("base")
    
    # Extract audio stream
    audio_path = extract_audio_stream(youtube_url)
    if not audio_path:
        print("Failed to extract audio stream.")
        return

    # Transcribe audio stream
    transcription = transcribe_stream(audio_path, model)

    # Save transcription to a text file
    with open(output_txt_path, 'w') as f:
        f.write(transcription)
        print(f"Transcription saved to {output_txt_path}")

    # Clean up temporary files
    os.remove(audio_path)
 
    
if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/watch?v=oyZY_BiTmd8"
    output_txt_path = "transcripts/transcription1.txt"
    main(youtube_url, output_txt_path)
