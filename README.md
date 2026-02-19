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
pip install fastapi uvicorn python-multipart
pip install pandas numpy scikit-learn
pip install pymupdf
pip install ultralytics opencv-python pillow
pip install gliner transformers sentence-transformers
pip install pytesseract pycountry
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
npx shadcn@latest init
npm install framer-motion
```

---

# 2 Download NLTK Stopwords (not relevante)

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
cd backend
python -m services.process_cvs_pipeline
```

Output:

* **CSV** → structured candidate data
* **JSON** → detected bounding boxes
* **PNG** → images of the detected bounding boxes divided by text_block, photo and skill_table


# 4 Run the program

## Backend (terminal 1)
```bash
cd backend
uvicorn app.main:app --reload
```


Open in browser:
```
http://127.0.0.1:8000/docs
```
Interactive Swagger UI will appear.
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

# 5 Features

* PDF → image conversion
* CV page detection
* Layout segmentation with YOLO
* OCR text extraction
* Entity recognition (skills, education, experience, contacts)
* CSV/JSON export
* Web interface for upload and visualization

---

# 6 License

Educational and research use.
