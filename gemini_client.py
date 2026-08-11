import os
import google.generativeai as genai

def generate_ai_response(query: str, title: str, author: str, lyrics: str, chords: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is not set in the Python environment."
    
    genai.configure(api_key=api_key)
    
    # We use a fast model since audio is already transcribed
    model = genai.GenerativeModel('gemini-3.6-flash')
    
    system_instruction = f"""
Bạn là ChordLive AI, chuyên gia âm nhạc. Hệ thống âm thanh DSP vừa bóc tách bài hát và cung cấp cho bạn 2 dữ liệu theo dòng thời gian (timeline) cực kỳ chính xác.
Thông tin bài hát:
- Tên: {title}
- Ca sĩ: {author}

--- TIMELINE LỜI BÀI HÁT ---
{lyrics}

--- TIMELINE HỢP ÂM (Dò bằng Librosa Chroma) ---
{chords}

Yêu cầu của người dùng là: {query}

Hãy dựa vào 2 timeline phía trên để trả lời người dùng. Nếu người dùng yêu cầu viết lời kèm hợp âm:
1. Bạn HÃY ĐỐI CHIẾU mốc thời gian của từng từ trong Lời Bài Hát với mốc thời gian của Hợp Âm để quyết định chính xác vị trí đặt Hợp Âm.
2. Viết dưới dạng Markdown Tab:
```tab
[C] Hôm [Am] qua em đi [F] tỉnh về [G]
```
3. CHÚ Ý: Bỏ qua các hợp âm nhiễu nếu chúng xuất hiện quá ngắn và không hợp lý. Hãy tinh chỉnh vòng hợp âm cho nghe lọt tai dựa trên kiến thức âm nhạc của bạn (vd: C F G Am).
"""
    
    response = model.generate_content(system_instruction)
    return response.text

def generate_text_response(query: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is not set in the Python environment."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash')
    
    system_instruction = f"""
Bạn là ChordLive AI, chuyên gia âm nhạc. Bạn luôn trả lời thân thiện, hữu ích.
Yêu cầu của người dùng là: {query}
"""
    
    response = model.generate_content(system_instruction)
    return response.text
