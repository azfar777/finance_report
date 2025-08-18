from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
import os, shutil, asyncio, subprocess

from generator import generate_decision_brief

app = FastAPI(title="RAG Financial Assistant")
DATA_DIR = os.getenv("DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html><body style="font-family: sans-serif; max-width: 800px; margin: 2rem auto">
      <h2>RAG Financial Assistant</h2>
      <form action="/upload" method="post" enctype="multipart/form-data">
        <p><b>Upload Annual Report (PDF)</b></p>
        <input type="file" name="file" accept="application/pdf"/>
        <button type="submit">Upload & Rebuild Index</button>
      </form>
      <hr/>
      <form action="/analyze" method="post">
        <p><b>Ticker</b> <input name="ticker" placeholder="AAPL or SU.TO"/></p>
        <button type="submit">Run Analysis</button>
      </form>
    </body></html>"""

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Save PDF
    dest = os.path.join(DATA_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Rebuild index by calling ingest.py as a subprocess (simple/robust)
    subprocess.run(["python", "ingest.py"], check=True)
    return {"status": "ok", "saved": file.filename}

@app.post("/analyze")
async def analyze(ticker: str = Form(...)):
    try:
        result = generate_decision_brief(ticker)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
