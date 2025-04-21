# 🧠 Multi-Modal RAG System

A versatile, lightweight Retrieval-Augmented Generation (RAG) system designed to handle all major content formats — videos, audio, images, text, PDFs, DOCX files, and URLs — making your entire digital library searchable and chat-friendly. Powered by OpenAI’s GPT models, this tool allows users to interact with their data using natural language queries.

Whether it’s a PDF report, a recorded interview, a presentation screenshot, or even a YouTube video — this system extracts relevant content, indexes it for fast retrieval, and returns intelligent, grounded responses with source references.
🌟 What Makes This Special?

## ✅ Universal Input Support: Works with almost any type of content — not just documents, but also images, audio, videos, web links, and more.

🎙️ Audio/Video Understanding: Uses Whisper to extract insights from spoken content in interviews, lectures, or podcasts, and makes them queryable just like documents.

🖼️ Image Intelligence: OCR-based extraction lets you search and question scanned documents, charts, presentations, and screenshots.

📄 URL Parsing (Coming Soon): Automatically extracts and embeds content from web pages using their URLs.

⚡ Fast Retrieval + GPT Reasoning: Combines FAISS vector search with OpenAI GPT to return fast, relevant, and explainable answers — grounded in your own data.

💻 Minimal Setup, Modular Design: Built to be easy to extend or plug into larger systems — great for hobbyists and production users alike.

🧠 LLM-Powered QA Interface: A polished Streamlit app allows you to upload files, ask questions, and browse answers — all in one place.

## 🚀 Features

- 📄 Extracts content from **PDFs**, **Word documents**, **images**, and **videos**
- 🧠 Transcribes audio using **Whisper** for video inputs
- 🧬 Embeds text using **Sentence-Transformers**
- 📦 Stores vectors using **FAISS** for fast retrieval
- 💬 Generates answers using **GPT-4**
- 🖥️ Clean **Streamlit UI** for asking questions and viewing sources

---

## 🛠️ Tech Stack

| Task                        | Library                     |
|----------------------------|-----------------------------|
| Text extraction (PDF/DOCX) | PyMuPDF, python-docx        |
| Image OCR                  | pytesseract, Pillow         |
| Video/audio transcription  | OpenCV, Whisper             |
| Embedding                  | sentence-transformers       |
| Vector DB                  | FAISS                       |
| LLM                        | OpenAI GPT (via API)        |
| UI                         | Streamlit                   |

---

## 📂 Project Structure

```bash

├── app/
│   ├── ingestion.py      # Extracts text/audio from files
│   ├── embedder.py       # Embeds documents into vectors
│   ├── retriever.py      # FAISS-based vector store
│   ├── generator.py      # Prompt building + GPT query
│   └── ui.py             # Streamlit frontend
├── data/                 # Your document uploads
├── vector_db/            # Saved FAISS index and metadata
├── build_index.py        # Script to build FAISS index
├── requirements.txt      # Python dependencies
└── README.md


✅ Getting Started

1. Clone the Repository

git clone https://github.com/Abhi21sar/Multi-Modal-RAG-System.git
cd Multi-Modal-RAG-System

2. Install Requirements

It’s recommended to use a virtual environment.

pip install -r requirements.txt

3. Add Your Documents

Place all your PDF, DOCX, image (.jpg, .png), or video (.mp4, .mov) files inside the data/ folder.

⸻

🏗️ Build the Vector Index

python build_index.py

This will:
	•	Extract and parse all supported files
	•	Generate embeddings using sentence-transformers
	•	Store them in a FAISS index under vector_db/

⸻

💬 Launch the Assistant

streamlit run app/ui.py

This opens a web interface where you can:
	•	Ask questions about your uploaded files
	•	See the AI-generated response
	•	Expand to view the supporting document excerpts

⸻

🔑 Setup OpenAI API

In app/generator.py, replace:

client = OpenAI(api_key="Your_OPENAI_API_KEY")

with your actual OpenAI API key.

⸻

✍️ Example Use Cases
	•	📝 “Summarize this 100-page research paper PDF.”
	•	🖼️ “What’s mentioned on the whiteboard in this image?”
	•	🎙️ “What were the key points discussed in this podcast?”
	•	🎞️ “Extract action items from the Zoom meeting recording.”
	•	🌐 “Give a summary of the article at this URL.” (coming soon)

⸻

📌 Roadmap & Future Improvements
	•	🧠 Add image embeddings via OpenAI CLIP
	•	🔗 RAG orchestration with LangChain
	•	🌐 URL crawling and semantic parsing
	•	🧩 Hybrid search: vector + keyword
	•	☁️ One-click Streamlit Cloud deployment

⸻

🤝 Contributing

Pull requests are welcome! If you have ideas to improve multi-modal handling or LLM prompting, feel free to open an issue or PR.

⸻

📄 License

MIT License

⸻

🙋‍♂️ Author

Abhishek Gurjar
GitHub: @Abhi21sar
Happy shipping! 🚀
