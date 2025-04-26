import os
import json
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {OPENAI_KEY}",
    "Content-Type": "application/json"
}

class AiManager:
    def __init__(self):
        self.base_url = "https://api.openai.com/v1"

    def summarize_and_store(self, previous_summary: str, user_prompt: str, assistant_response: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a summarizer AI..."
                )
            },
            {
                "role": "user",
                "content": f"Previous summary:\n{previous_summary}\n\nNew message:\nUser: {user_prompt}\nChibi: {assistant_response}"
            }
        ]
        body = {"model": "gpt-4", "messages": messages}
        res = requests.post(f"{self.base_url}/chat/completions", headers=HEADERS, json=body)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

    def send_chat(self, prompt: str, summary: str = "") -> str:
        memory = f'Use this memory:\n"{summary}". ' if summary else ""
        messages = [
            {"role": "system", "content": memory + "You are Chibi..."},
            {"role": "user", "content": prompt}
        ]
        body = {"model": "gpt-4", "messages": messages}
        res = requests.post(f"{self.base_url}/chat/completions", headers=HEADERS, json=body)
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]

    def stt(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            files = {
                "file": ("audio.wav", f, "audio/wav"),
            }
            data = {
                "model": "whisper-1",
                "temperature": 1.0,
                "response_format": "json",
                "language": "en"
            }
            res = requests.post(f"{self.base_url}/audio/transcriptions", headers={"Authorization": f"Bearer {OPENAI_KEY}"}, files=files, data=data)
            res.raise_for_status()
            return res.json()["text"]

    def tts(self, text: str) -> bytes:
        body = {
            "model": "tts-1",
            "input": text,
            "voice": "nova"
        }
        res = requests.post(f"{self.base_url}/audio/speech", headers=HEADERS, json=body)
        res.raise_for_status()
        return res.content

    def generate_image(self, prompt: str) -> str:
        full_prompt = (
            "Generate a 2D cartoon game environment...\n"
            f"Requested Environment: {prompt}"
        )
        body = {"prompt": full_prompt, "n": 1, "size": "1024x1024"}
        res = requests.post(f"{self.base_url}/images/generations", headers=HEADERS, json=body)
        res.raise_for_status()
        return res.json()["data"][0]["url"]
