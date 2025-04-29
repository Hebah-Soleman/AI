from fastapi import FastAPI, File, UploadFile, Form, Request
from pydantic import BaseModel
from service.ai_manager import AiManager
import base64
from fastapi import Form
from fastapi.responses import JSONResponse

app = FastAPI()
ai = AiManager()

class Prompt(BaseModel):
    prompt: str
    user: str = ""

@app.post("/chat/")
def chat(data: Prompt):
    response = ai.send_chat(data.prompt, data.user)
    return {"response": response}

@app.post("/summarize/")
def summarize(data: Prompt):
    summary = ai.summarize_and_store(data.user, data.prompt, "Chibi’s response here...")
    return {"summary": summary}

@app.post("/tts/")
def speech(text: str = Form(...)):
    mp3 = ai.tts(text)
    if mp3 is None:
        return JSONResponse(status_code=500, content={"error": "Audio generation failed"})

    # Convert to base64 string
    b64_audio = base64.b64encode(mp3).decode("utf-8")
    return {"base64_audio": b64_audio}

@app.post("/stt/")
def stt(file: UploadFile = File(...)):
    path = f".\{file.filename}"
    with open(path, "wb") as f:
        f.write(file.file.read())
    text = ai.stt(path)
    return {"transcription": text}

@app.post("/image/")
def image(prompt: str = Form(...)):
    url = ai.generate_image(prompt)
    return {"image_url": url}