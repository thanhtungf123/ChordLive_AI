from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import sys


app = FastAPI(title="ChordLive AI Audio Service")

class ChatRequest(BaseModel):
    messages: list[dict]
    youtube_url: str | None = None

@app.post("/api/analyze")
async def analyze_audio(request: ChatRequest):
    from gemini_client import generate_ai_response_with_audio, generate_text_response
    
    if not request.youtube_url:
        return {"response": generate_text_response(request.messages)}
    
    try:
        from audio_processor import download_audio
        import os
        
        # 1. Download
        audio_path, title, author = download_audio(request.youtube_url)
        
        # 2. Synthesize response using Gemini with audio file attached
        final_text = generate_ai_response_with_audio(request.messages, title, author, audio_path)
        
        # Cleanup original audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            
        return {
            "response": final_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
