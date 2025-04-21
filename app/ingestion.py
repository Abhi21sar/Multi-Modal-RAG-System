# app/ingestion.py

import os
import fitz  # PyMuPDF
import certifi
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context
#import pymupdf  # PyMuPDF
import docx
import pytesseract
from PIL import Image
import cv2
import whisper

TEXT_EXT = [".pdf", ".docx"]
IMAGE_EXT = [".png", ".jpg", ".jpeg"]
VIDEO_EXT = [".mp4", ".mov", ".avi"]

whisper_model = whisper.load_model("base")  # or "small", "medium", "large"

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(filepath):
    doc = docx.Document(filepath)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_image(filepath):
    image = Image.open(filepath)
    return pytesseract.image_to_string(image)

def extract_audio_from_video(filepath, output_wav="temp.wav"):
    video = cv2.VideoCapture(filepath)
    # Save audio using ffmpeg (requires ffmpeg installed)
    output_wav = filepath.replace(".mp4", ".wav")
    os.system(f"ffmpeg -y -i \"{filepath}\" -vn -acodec pcm_s16le -ar 16000 -ac 1 \"{output_wav}\"")
    return output_wav

def transcribe_audio(filepath):
    result = whisper_model.transcribe(filepath)
    return result['text']

def extract_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext in IMAGE_EXT:
        return extract_text_from_image(filepath)
    elif ext in VIDEO_EXT:
        audio_path = extract_audio_from_video(filepath)
        return transcribe_audio(audio_path)
    else:
        return ""

def extract_from_folder(folder_path):
    knowledge_base = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            content = extract_from_file(full_path)
            if content.strip():
                knowledge_base.append({
                    "filename": file,
                    "filepath": full_path,
                    "content": content
                })
    return knowledge_base