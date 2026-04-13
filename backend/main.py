from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber

app = FastAPI()

# =========================
# 🔥 CORS CONFIG
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🧠 AI MODEL (EMBEDDINGS)
# =========================
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# 📦 REQUEST MODEL
# =========================
class MatchRequest(BaseModel):
    cv: str
    job: str


# =========================
# 🧠 AI SIMILARITY (NEW)
# =========================
def compute_similarity(cv_text: str, job_text: str) -> float:
    embeddings = model.encode([cv_text, job_text])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])
    return float(score[0][0])


# =========================
# 🔥 SIMPLE ROOT
# =========================
@app.get("/")
def root():
    return {"message": "CV Job Matcher AI running 🚀"}


# =========================
# 🔵 MATCH ENDPOINT (AI)
# =========================
@app.post("/match")
def match(data: MatchRequest):
    score = compute_similarity(data.cv, data.job)

    if score > 0.75:
        level = "High"
    elif score > 0.45:
        level = "Medium"
    else:
        level = "Low"

    return {
        "score": round(score * 100, 2),
        "match_level": level
    }


# =========================
# 🔥 FULL AI ANALYSIS
# =========================
@app.post("/analyze")
def analyze(data: MatchRequest):
    score = compute_similarity(data.cv, data.job)

    if score > 0.75:
        level = "High"
    elif score > 0.45:
        level = "Medium"
    else:
        level = "Low"

    # =========================
    # 🧠 AI REMARKS (SMART)
    # =========================
    if score > 0.75:
        remarks = "Excellent match. Candidate fits very well the job requirements."
    elif score > 0.45:
        remarks = "Good match but some skills are missing or weakly represented."
    else:
        remarks = "Low match. Candidate profile is not aligned with job requirements."

    return {
        "score": round(score * 100, 2),
        "match_level": level,
        "remarks": remarks
    }


# =========================
# 📄 UPLOAD PDF CV
# =========================
@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):

    text = ""

    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

    return {
        "cv_text": text.strip()
    }