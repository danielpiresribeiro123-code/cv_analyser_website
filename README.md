Aqui tens um **README simples, limpo e completo**, pronto a colar no teu repositório.

---

# CV Analyser Website

End-to-end system for **automatic Curriculum Vitae processing**, combining:

* Computer Vision (YOLO layout detection)
* OCR text extraction
* Named Entity Recognition (GLiNER + regex)
* FastAPI backend + React frontend

The system converts **PDF CVs → structured candidate data**.

---

# Project Structure

```
cv_analyser_website/
│
├── backend/
│   ├── app/            # FastAPI API
│   ├── pipeline/       # CV processing pipeline (YOLO + OCR + NLP)
│   ├── models/         # ML models (YOLO, OCR, etc.)
│   ├── uploads/        # Input CVs
│   ├── outputs/        # Generated CSV / JSON
│   └── requirements.txt
│
├── frontend/           # React + Vite interface
│
└── README.md
```

---

# 1 Install Dependencies

## Backend

Inside **backend/** run:

```bash
pip install fastapi uvicorn python-multipart \
pandas numpy scikit-learn \
pymupdf \
ultralytics opencv-python pillow \
gliner transformers sentence-transformers \
pytesseract
```

Extra libraries used internally:

* **langchain-community** → PDF loader
* **pymupdf** → PDF reading
* **nltk** → stopwords
* **transformers + torch** → language models

---

## Frontend

Inside **frontend/** run:

```bash
npm install
npx tailwindcss init -p   # if config does not exist
npm install react react-dom
npm install -D @vitejs/plugin-react
```

---

# 2 Download NLTK Stopwords (one time)

```bash
python -m nltk.downloader stopwords
```

or:

```bash
python pipeline/fast_analysis/modules/install_stop_words.py
```

---

# 3 Test the CV Pipeline

Run from project root:

```bash
python -m backend.pipeline.pipelines.run_test_process_single_cv \
backend/uploads/test.pdf \
outputs/out.csv \
outputs/out.json
```

Output:

* **CSV** → structured candidate data
* **JSON** → detected bounding boxes

---

# 4 Run the API

Inside **backend/**:

```bash
uvicorn app.main:app --reload
```

Open in browser:

```
http://127.0.0.1:8000/docs
```

Interactive Swagger UI will appear.

---

# 5 Run the Frontend

## Backend (terminal 1)

```bash
cd backend
uvicorn app.main:app --reload
```

## Frontend (terminal 2)

```bash
cd frontend
npm run dev
```

Open:

```
http://localhost:5173
```

---

# 6 Features

* PDF → image conversion
* CV page detection
* Layout segmentation with YOLO
* OCR text extraction
* Entity recognition (skills, education, experience, contacts)
* CSV/JSON export
* Web interface for upload and visualization

---

# 7 License

Educational and research use.

---

Se quiseres, no próximo passo posso fazer:

* **README académico estilo tese**
* **diagrama do pipeline**
* **GIF de demonstração**
* **badges profissionais**

Isso é o que realmente transforma isto num **portefólio forte**.
