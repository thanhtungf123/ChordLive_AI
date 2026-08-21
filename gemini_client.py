import os
import google.generativeai as genai

def generate_ai_response(messages: list, title: str, author: str, lyrics: str, chords: str) -> str:
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

Dưới đây là lịch sử trò chuyện của bạn với người dùng:

Hãy dựa vào 2 timeline phía trên để trả lời người dùng. Nếu người dùng yêu cầu viết lời kèm hợp âm:
1. Bạn HÃY ĐỐI CHIẾU mốc thời gian của từng từ trong Lời Bài Hát với mốc thời gian của Hợp Âm để quyết định chính xác vị trí đặt Hợp Âm.
2. Viết dưới dạng Markdown Tab:
```tab
[C] Hôm [Am] qua em đi [F] tỉnh về [G]
```
```
3. CHÚ Ý: Bỏ qua các hợp âm nhiễu nếu chúng xuất hiện quá ngắn và không hợp lý. Hãy tinh chỉnh vòng hợp âm cho nghe lọt tai dựa trên kiến thức âm nhạc của bạn (vd: C F G Am).
"""
    for msg in messages:
        role_name = "Người dùng" if msg["role"] == "user" else "AI"
        system_instruction += f"{role_name}: {msg['content']}\n\n"
        
    system_instruction += "AI (hãy trả lời tiếp):"
    
    response = model.generate_content(system_instruction)
    return response.text

def generate_text_response(messages: list) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is not set in the Python environment."
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.6-flash')
    
    system_instruction = f"""
Bạn là ChordLive AI, chuyên gia âm nhạc. Bạn luôn trả lời thân thiện, hữu ích. Dưới đây là lịch sử trò chuyện:

"""
    for msg in messages:
        role_name = "Người dùng" if msg["role"] == "user" else "AI"
        system_instruction += f"{role_name}: {msg['content']}\n\n"
        
    system_instruction += "AI (hãy trả lời tiếp):"
    
    response = model.generate_content(system_instruction)
    return response.text
