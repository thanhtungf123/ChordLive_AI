import os
import google.generativeai as genai

def generate_ai_response_with_audio(messages: list, title: str, author: str, audio_path: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is not set in the Python environment."
    
    genai.configure(api_key=api_key)
    
    # Upload the audio file to Gemini
    audio_file = genai.upload_file(path=audio_path)
    
    # Use Gemini 1.5 Flash which has native multimodal audio support
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    system_instruction = f"""
Bạn là ChordLive AI, chuyên gia âm nhạc siêu đẳng. 
Thông tin bài hát:
- Tên: {title}
- Ca sĩ: {author}

Tệp âm thanh đính kèm chính là bài hát này. Nhiệm vụ của bạn là:
1. NGHE bài hát và BÓC TÁCH LỜI (lyrics).
2. TÌM HỢP ÂM (chords) chính xác cho từng câu hát.
3. Nếu người dùng yêu cầu, hãy viết lại lời bài hát kèm hợp âm lồng vào trong ngoặc vuông ngay trước từ được chuyển hợp âm.

Định dạng Markdown Tab bắt buộc:
```tab
[C] Hôm [Am] qua em đi [F] tỉnh về [G]
```

Chú ý:
- Vòng hợp âm phải thật mượt mà, đúng chuẩn nhạc lý (vd: Tone C thì thường có C, F, G, Am...).
- Đừng bịa lời nếu không nghe rõ, hãy suy luận theo ngữ cảnh bài hát.
"""
    
    chat_history = [
        {"role": "user", "parts": [system_instruction]}
    ]
    
    for msg in messages:
        # Convert internal role to gemini role
        role = "user" if msg["role"] == "user" else "model"
        chat_history.append({"role": role, "parts": [msg["content"]]})
        
    # We pass the audio_file along with the last user message
    # To do this correctly in Gemini API, we can just append it to the last user message's parts.
    if chat_history[-1]["role"] == "user":
        chat_history[-1]["parts"].insert(0, audio_file)
    else:
        # If the last message wasn't user (unlikely), add a new user message with the file
        chat_history.append({"role": "user", "parts": [audio_file, "Hãy phân tích bài hát này theo yêu cầu trước đó."]})
    
    # Initialize chat session
    # Note: We just send everything at once in contents.
    response = model.generate_content(chat_history)
    
    # Delete the file from Gemini storage to save space
    genai.delete_file(audio_file.name)
    
    return response.text

def generate_text_response(messages: list) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is not set in the Python environment."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    chat_history = [
        {"role": "user", "parts": ["Bạn là ChordLive AI, chuyên gia âm nhạc. Bạn luôn trả lời thân thiện, hữu ích."]}
    ]
    
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        chat_history.append({"role": role, "parts": [msg["content"]]})
        
    response = model.generate_content(chat_history)
    return response.text
