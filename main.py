from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys

# Ensure ffmpeg in current directory is found by Whisper and yt-dlp
os.environ["PATH"] = os.path.dirname(os.path.abspath(__file__)) + os.pathsep + os.environ["PATH"]

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../backend/.env'), override=True)

from fastapi import FastAPI, HTTPException
app = FastAPI(title="ChordLive AI Audio Service")

class ChatRequest(BaseModel):
    messages: list[dict]
    youtube_url: str | None = None

@app.post("/api/analyze")
async def analyze_audio(request: ChatRequest):
    from gemini_client import generate_ai_response, generate_text_response
    
    if not request.youtube_url:
        # Handle normal text chat without audio
        return {"response": generate_text_response(request.messages)}
    
    try:
        from audio_processor import download_audio, extract_lyrics, extract_chords
        import os
        
        # 1. Download
        audio_path, title, author = download_audio(request.youtube_url)
        
        # 2. Separate Audio (Demucs)
        from audio_processor import separate_audio
        tracks = separate_audio(audio_path)
        
        # 3. Extract
        lyrics_segments = extract_lyrics(tracks["vocals"])
        chord_segments = extract_chords(tracks["bass"], tracks["other"])
        
        # Format Timelines
        lyrics_timeline = ""
        for seg in lyrics_segments:
            lyrics_timeline += f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}\n"
            
        chords_timeline = ""
        for seg in chord_segments:
            chords_timeline += f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['chord']}\n"
        
        # Cleanup original and separated files
        if os.path.exists(audio_path):
            os.remove(audio_path)
        import shutil
        if os.path.exists("separated"):
            shutil.rmtree("separated")
            
        # 4. Synthesize response using Gemini
        from gemini_client import generate_ai_response
        final_text = generate_ai_response(request.messages, title, author, lyrics_timeline, chords_timeline)
        
        return {
            "response": final_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
