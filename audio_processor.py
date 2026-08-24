import os
import yt_dlp

def download_audio(youtube_url: str):
    options = {
        'format': 'bestaudio/best',
        'outtmpl': '%(id)s.%(ext)s',
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
