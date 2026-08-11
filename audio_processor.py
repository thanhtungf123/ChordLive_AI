import os
import yt_dlp
import whisper
import librosa
import numpy as np
import subprocess
import shutil

# Tải Whisper model (dùng bản 'base' để chạy mượt trên CPU)
print("Loading Whisper model (base)...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")

# Define major and minor chord templates
chords = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
chord_templates = []
chord_labels = []

for i in range(12):
    # Major chord
    template = np.zeros(12)
    template[0] = 1
    template[4] = 1
    template[7] = 1
    template = np.roll(template, i)
    chord_templates.append(template)
    chord_labels.append(chords[i])
    
    # Minor chord
    template = np.zeros(12)
    template[0] = 1
    template[3] = 1
    template[7] = 1
    template = np.roll(template, i)
    chord_templates.append(template)
    chord_labels.append(chords[i] + 'm')

chord_templates = np.array(chord_templates)

def download_audio(youtube_url: str):
    options = {
        'format': 'bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
        'ffmpeg_location': '.',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'noplaylist': True,
        'quiet': True
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = f"{info['id']}.mp3"
        return filename, info.get('title', 'Unknown Title'), info.get('uploader', 'Unknown Artist')

def separate_audio(audio_path: str) -> dict:
    print(f"Separating sources for {audio_path} using Demucs...")
    out_dir = "separated"
    # Gọi demucs bằng subprocess để tránh lỗi xung đột đa luồng trong FastAPI
    subprocess.run(["demucs", audio_path, "-o", out_dir, "-n", "htdemucs"], check=True)
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    track_dir = os.path.join(out_dir, "htdemucs", base_name)
    
    return {
        "vocals": os.path.join(track_dir, "vocals.wav"),
        "bass": os.path.join(track_dir, "bass.wav"),
        "drums": os.path.join(track_dir, "drums.wav"),
        "other": os.path.join(track_dir, "other.wav")
    }

def extract_lyrics(vocals_path: str) -> list:
    print(f"Extracting lyrics from {vocals_path}...")
    result = whisper_model.transcribe(vocals_path, fp16=False)
    
    segments = []
    for seg in result['segments']:
        segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': seg['text'].strip()
        })
    return segments

def extract_chords(bass_path: str, other_path: str) -> str:
    print(f"Extracting chords from {bass_path} and {other_path}...")
    
    # Tải 2 track bass và other (đã loại bỏ tiếng trống và giọng hát)
    y_bass, sr = librosa.load(bass_path)
    y_other, _ = librosa.load(other_path, sr=sr)
    
    # Trộn 2 track lại làm Backing Track thuần hợp âm
    # Đảm bảo độ dài bằng nhau
    min_len = min(len(y_bass), len(y_other))
    y_mixed = y_bass[:min_len] + y_other[:min_len]
    
    # Phân tích nhịp điệu (Beat tracking)
    tempo, beats = librosa.beat.beat_track(y=y_mixed, sr=sr)
    
    # Trích xuất biểu đồ âm sắc (Chroma)
    chroma = librosa.feature.chroma_cqt(y=y_mixed, sr=sr)
    
    # Đồng bộ biểu đồ âm sắc theo từng nhịp (Beat-synchronous) để triệt tiêu nhiễu
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
    
    # Chuẩn hóa
    chroma_sync = librosa.util.normalize(chroma_sync, norm=2, axis=0)
    
    # Tính toán độ tương đồng với các mẫu hợp âm
    similarities = np.dot(chord_templates, chroma_sync)
    
    # Lấy thời gian của từng nhịp
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Lấy hợp âm tốt nhất cho từng phách (beat)
    best_chords_idx = np.argmax(similarities, axis=0)
    
    # Tạo danh sách hợp âm kèm timestamp, gộp các hợp âm giống nhau liên tiếp
    chord_segments = []
    last_chord = None
    start_time = 0.0
    
    for i, idx in enumerate(best_chords_idx):
        chord = chord_labels[idx]
        current_time = beat_times[i] if i < len(beat_times) else start_time
        
        if chord != last_chord:
            if last_chord is not None:
                chord_segments.append({
                    'start': start_time,
                    'end': current_time,
                    'chord': last_chord
                })
            start_time = current_time
            last_chord = chord
            
    # Add the final chord
    if last_chord is not None:
        chord_segments.append({
            'start': start_time,
            'end': beat_times[-1] if len(beat_times) > 0 else start_time + 1.0,
            'chord': last_chord
        })
        
    return chord_segments
