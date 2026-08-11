@echo off
call venv\Scripts\activate
echo Starting FastAPI server...
uvicorn main:app --reload
