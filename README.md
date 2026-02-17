### Primeiro faz download das bibliotecas:
Dentro da pasta backend executa:
```
pip install fastapi uvicorn python-multipart \
pandas numpy scikit-learn \
pymupdf \
ultralytics opencv-python pillow \
gliner transformers sentence-transformers \
pytesseract

```
* langchain-community → loader do PDF
* pymupdf → leitura de PDFs
* nltk → stopwords
* transformers + torch → modelo BERT

Dentro da pasta frontend executa:
```
npm install
npx tailwindcss init -p (se não existir)
npm install react react-dom
npm install -D @vitejs/plugin-react (plugin react do vite)
```


### Testar code:
```
python -m backend.pipeline.pipelines.run_test_process_single_cv backend/uploads/test.pdf outputs/out.csv outputs/out.json
```


### Caso código precisa das stopwords:

Executa uma vez:
```
python -m nltk.downloader stopwords
```
ou:
```
python pipeline/fast_analysis/modules/install_stop_words.py
```



### Abrir site exemplo simples/Correr Api:
#### Dentro de backend/:
```
uvicorn app.main:app --reload
```

#### Abrir no browser:
```
http://127.0.0.1:8000/docs
```



Backend
cd backend
uvicorn app.main:app --reload

Frontend
cd frontend
npm run dev


Abrir:

http://localhost:5173