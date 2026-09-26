from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import joblib
import json
import os
import bcrypt
import jwt
import datetime
from sqlalchemy.orm import Session
from io import BytesIO

import uuid
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import local modules
from backend.database import SessionLocal, User, Assessment, Hospital, PeriodEntry, ConversationSession, ChatMessage, HealthJournalEntry, init_db
from backend.openai_service import run_openai_chat
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# Initialize database
init_db()

app = FastAPI(title="PCOS/PCOD Early Detection System API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("JWT_SECRET", "super_secret_pcos_key_123_secure_32bytes_key")
ALGORITHM = "HS256"

# Load Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic schemas
class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class ClinicalInputs(BaseModel):
    age: float
    height: float
    weight: float
    cycleLength: float
    cyclePeriod: float
    bloodGlucose: Optional[float] = None
    sleepDuration: float
    familyHistory: bool
    physicalActivity: int # 1, 2, 3
    stressLevel: int # 1 to 10
    fsh: Optional[float] = None
    lh: Optional[float] = None
    amh: Optional[float] = None
    testosterone: Optional[float] = None
    fastingInsulin: Optional[float] = None

class Symptoms(BaseModel):
    irregularPeriods: bool
    noPeriods: bool
    excessHair: bool
    acne: bool
    hairLoss: bool
    darkPatches: bool
    weightGain: bool
    difficultyLosingWeight: bool
    fatigue: bool
    moodSwings: bool
    pelvicPain: bool
    infertility: bool

class PredictionRequest(BaseModel):
    clinicalInputs: ClinicalInputs
    symptoms: Symptoms

class PeriodEntryCreate(BaseModel):
    startDate: str
    endDate: str
    notes: Optional[str] = None
    durationDays: Optional[int] = 1

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    history: List[Dict[str, str]] = []

class JournalEntryCreate(BaseModel):
    entryDate: Optional[str] = None
    freeText: Optional[str] = ""
    selectedSymptoms: Optional[List[str]] = []
    tags: Optional[List[str]] = []
    consentForAI: Optional[bool] = True

class JournalEntryUpdate(BaseModel):
    freeText: Optional[str] = None
    selectedSymptoms: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    consentForAI: Optional[bool] = None

class ConsentUpdate(BaseModel):
    consentForAI: bool


# Auth helper functions
def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(days=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str, db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# --- Auth Endpoints ---
@app.post("/api/auth/register")
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user_data.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed = get_password_hash(user_data.password)
    new_user = User(email=user_data.email, hashed_password=hashed)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    token = create_access_token(data={"sub": new_user.email})
    return {"token": token, "email": new_user.email}

@app.post("/api/auth/login")
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    token = create_access_token(data={"sub": user.email})
    return {"token": token, "email": user.email}

# --- Hospital Finder Endpoints ---
@app.get("/api/hospitals")
def get_hospitals(state: Optional[str] = None, district: Optional[str] = None, db: Session = Depends(get_db)):
    query = db.query(Hospital)
    if state:
        query = query.filter(Hospital.state.ilike(state))
    if district:
        query = query.filter(Hospital.district.ilike(district))
    return query.all()

@app.get("/api/hospitals/regions")
def get_regions(db: Session = Depends(get_db)):
    # Returns unique states and their corresponding districts
    hospitals = db.query(Hospital).all()
    regions = {}
    for h in hospitals:
        if h.state not in regions:
            regions[h.state] = set()
        regions[h.state].add(h.district)
    
    # Format for JSON response
    return {state: sorted(list(districts)) for state, districts in regions.items()}

# --- ML Prediction Logic & Endpoints ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_ml_assets():
    try:
        model_path = os.path.join(BASE_DIR, 'best_model.joblib')
        scaler_path = os.path.join(BASE_DIR, 'scaler.joblib')
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            parent_dir = os.path.dirname(BASE_DIR)
            model_path = os.path.join(parent_dir, 'backend', 'best_model.joblib')
            scaler_path = os.path.join(parent_dir, 'backend', 'scaler.joblib')
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading ML models: {e}")
        return None, None

@app.post("/api/predict")
def predict_pcod(req: PredictionRequest, authorization: Optional[str] = None, db: Session = Depends(get_db)):
    model, scaler = load_ml_assets()
    
    c = req.clinicalInputs
    s = req.symptoms
    
    # Authenticate user if token is provided
    current_user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            current_user = get_current_user(token, db)
        except HTTPException:
            pass
            
    # Compute derived variables
    bmi_val = c.weight / ((c.height / 100) ** 2)
    cycle_reg = 1 if (s.irregularPeriods or s.noPeriods) else 0
    
    # Prepare features matching model feature names
    # Age, Height, Weight, BMI, CycleRegularity, CycleLength, CyclePeriod, Hirsutism, Acne, HairLoss,
    # DarkPatches, WeightGain, DifficultyLosingWeight, Fatigue, MoodSwings, PelvicPain, Infertility,
    # FamilyHistory, PhysicalActivity, StressLevel, SleepDuration, FSH, LH, AMH, Testosterone, FastingInsulin, BloodGlucose
    
    # Fill optional parameters with fallback averages if not supplied
    blood_glucose = c.bloodGlucose if c.bloodGlucose is not None else 90.0
    fsh = c.fsh if c.fsh is not None else 5.5
    lh = c.lh if c.lh is not None else 6.0
    amh = c.amh if c.amh is not None else 2.5
    testosterone = c.testosterone if c.testosterone is not None else 35.0
    fasting_insulin = c.fastingInsulin if c.fastingInsulin is not None else 8.0
    
    feature_dict = {
        'Age': c.age,
        'Height': c.height,
        'Weight': c.weight,
        'BMI': bmi_val,
        'CycleRegularity': cycle_reg,
        'CycleLength': c.cycleLength,
        'CyclePeriod': c.cyclePeriod,
        'Hirsutism': int(s.excessHair),
        'Acne': int(s.acne),
        'HairLoss': int(s.hairLoss),
        'DarkPatches': int(s.darkPatches),
        'WeightGain': int(s.weightGain),
        'DifficultyLosingWeight': int(s.difficultyLosingWeight),
        'Fatigue': int(s.fatigue),
        'MoodSwings': int(s.moodSwings),
        'PelvicPain': int(s.pelvicPain),
        'Infertility': int(s.infertility),
        'FamilyHistory': int(req.clinicalInputs.familyHistory),
        'PhysicalActivity': c.physicalActivity,
        'StressLevel': c.stressLevel,
        'SleepDuration': c.sleepDuration,
        'FSH': fsh,
        'LH': lh,
        'AMH': amh,
        'Testosterone': testosterone,
        'FastingInsulin': fasting_insulin,
        'BloodGlucose': blood_glucose
    }
    
    features_df = pd.DataFrame([feature_dict])
    
    # Scale continuous columns
    continuous_cols = ['Age', 'Height', 'Weight', 'BMI', 'CycleLength', 'CyclePeriod', 'StressLevel', 'SleepDuration', 
                       'FSH', 'LH', 'AMH', 'Testosterone', 'FastingInsulin', 'BloodGlucose']
    
    features_scaled = features_df.copy()
    features_scaled[continuous_cols] = scaler.transform(features_df[continuous_cols])
    
    # Predict Probability
    prob = float(model.predict_proba(features_scaled)[0, 1])
    
    # Clinical evidence-based score (Rotterdam & AE-PCOS criteria)
    clinical_score = 0
    if s.irregularPeriods or s.noPeriods or c.cycleLength > 35 or c.cycleLength < 21:
        clinical_score += 30
        
    symptom_count = sum([
        int(s.excessHair), int(s.acne), int(s.hairLoss), int(s.darkPatches),
        int(s.weightGain), int(s.difficultyLosingWeight), int(s.fatigue),
        int(s.moodSwings), int(s.pelvicPain), int(s.infertility)
    ])
    clinical_score += min(45, symptom_count * 9)
    
    if bmi_val >= 25:
        clinical_score += 15
    if lh and fsh and fsh > 0 and (lh / fsh) >= 1.8:
        clinical_score += 15
    if testosterone and testosterone >= 45:
        clinical_score += 10
    if req.clinicalInputs.familyHistory:
        clinical_score += 10

    clinical_score = min(95, clinical_score)

    # Combined calibrated ensemble score
    ml_score = prob * 100
    risk_score = round(0.6 * ml_score + 0.4 * clinical_score)
    
    # Floor adjustment based on active clinical symptoms
    if symptom_count >= 2 and risk_score < 35:
        risk_score = 35 + (symptom_count * 5)
    elif symptom_count == 0 and not s.irregularPeriods and not s.noPeriods and bmi_val < 25:
        risk_score = min(risk_score, 15)

    risk_score = min(98, max(2, risk_score))
    
    # Risk Level mapping
    if risk_score <= 30:
        risk_level = "Low"
    elif risk_score <= 70:
        risk_level = "Moderate"
    else:
        risk_level = "High"
        
    # Calculate Prediction Confidence (distance from decision boundary 0.5)
    confidence = float(abs(prob - 0.5) * 2) # scaled to 0.0 - 1.0
    
    # Explainable AI: Feature Importance Contribution
    # To compute SHAP-like contributions locally without running expensive shap calculations on every request,
    # we determine feature contribution based on scaled distance from training average multiplied by feature weights/importances.
    # We retrieve the feature importances from the model.
    importances = None
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    
    # Fallback to general clinical importances if not found
    if importances is None:
        importances = np.ones(len(feature_dict)) / len(feature_dict)
        
    contribs = []
    features_list = list(feature_dict.keys())
    
    # Load dataset to get training means
    try:
        csv_path = os.path.join(BASE_DIR, 'pcod_dataset.csv')
        train_df = pd.read_csv(csv_path)
        train_means = train_df.mean().to_dict()
    except Exception:
        # Clinical normal means fallback
        train_means = {
            'Age': 26.0, 'Height': 160.0, 'Weight': 60.0, 'BMI': 23.4, 'CycleRegularity': 0.3,
            'CycleLength': 29.0, 'CyclePeriod': 5.0, 'Hirsutism': 0.15, 'Acne': 0.2, 'HairLoss': 0.15,
            'DarkPatches': 0.08, 'WeightGain': 0.25, 'DifficultyLosingWeight': 0.2, 'Fatigue': 0.3,
            'MoodSwings': 0.35, 'PelvicPain': 0.15, 'Infertility': 0.1, 'FamilyHistory': 0.15,
            'PhysicalActivity': 2.0, 'StressLevel': 5.0, 'SleepDuration': 7.2, 'FSH': 6.0, 'LH': 6.0,
            'AMH': 2.2, 'Testosterone': 35.0, 'FastingInsulin': 8.0, 'BloodGlucose': 90.0
        }
        
    for idx, feature in enumerate(features_list):
        feat_val = float(feature_dict[feature])
        mean_val = float(train_means.get(feature, feat_val))
        
        # Calculate contribution: direction (positive if values increase risk, negative if they decrease it)
        # For PCOS, higher values of most features (except FSH, Sleep, Activity) increase risk.
        pcos_direct_correlations = ['BMI', 'CycleLength', 'CycleRegularity', 'Hirsutism', 'Acne', 'HairLoss',
                                    'DarkPatches', 'WeightGain', 'DifficultyLosingWeight', 'Fatigue', 'MoodSwings',
                                    'PelvicPain', 'Infertility', 'FamilyHistory', 'StressLevel', 'LH', 'AMH',
                                    'Testosterone', 'FastingInsulin', 'BloodGlucose', 'Weight']
        
        direction = 1 if feature in pcos_direct_correlations else -1
        diff = feat_val - mean_val
        
        # Weight difference by feature importance
        contrib_val = diff * importances[idx] * direction
        
        # For symptoms, if it's true (1), it contributes directly
        if feature in [k for k in feature_dict if feature_dict[k] in [0, 1]] and feat_val == 1:
            contrib_val = importances[idx]
        elif feature in [k for k in feature_dict if feature_dict[k] in [0, 1]] and feat_val == 0:
            contrib_val = -0.2 * importances[idx]
            
        contribs.append({
            "feature": feature,
            "contribution": float(contrib_val),
            "value": feat_val
        })
        
    # Sort contributions by magnitude and keep top 5 indicators
    contribs_sorted = sorted(contribs, key=lambda x: abs(x['contribution']), reverse=True)[:5]
    
    # Generate recommendations
    recs = []
    if risk_level == "Low":
        recs = [
            "Maintain a balanced diet rich in whole grains, fruits, and vegetables to keep metabolic parameters stable.",
            "Incorporate at least 30 minutes of moderate physical activity daily.",
            "Continue tracking your menstrual cycle regularity and duration using a calendar app.",
            "Schedule regular annual check-ups with your primary care provider."
        ]
    elif risk_level == "Moderate":
        recs = [
            "Reduce intake of refined sugar, high-glycemic carbohydrates, and processed foods to manage insulin levels.",
            "Integrate strength training or resistance exercises 3 times a week to improve insulin sensitivity.",
            "Consider consult a gynecologist for a routine pelvic ultrasound and physical evaluation.",
            "Get a hormone blood profile (LH, FSH, AMH, Testosterone) to check clinical baseline metrics.",
            "Focus on stress reduction practices such as deep breathing, meditation, or yoga."
        ]
    else: # High
        recs = [
            "Schedule an appointment with a gynecologist or a reproductive endocrinologist as soon as possible.",
            "Get a comprehensive diagnostic workup, including a pelvic ultrasound to look for polycystic ovaries.",
            "Request a complete blood panel: Fasting Insulin, Fasting Glucose, AMH, FSH, LH, and Free Testosterone.",
            "Strictly limit processed sugars and refined carbohydrates, favoring a low-GI, high-fiber dietary pattern.",
            "Discuss clinical management options (like lifestyle therapy or metformin) with your doctor.",
            "Download this PDF assessment report and bring it to your clinical appointment."
        ]
        
    # Save to Database
    db_assessment = Assessment(
        user_id=current_user.id if current_user else None,
        age=c.age, height=c.height, weight=c.weight, bmi=bmi_val,
        cycle_regularity=cycle_reg, cycle_length=c.cycleLength, cycle_period=c.cyclePeriod,
        irregular_periods=s.irregularPeriods, no_periods=s.noPeriods, excess_hair=s.excessHair,
        acne=s.acne, hair_loss=s.hairLoss, dark_patches=s.darkPatches, weight_gain=s.weightGain,
        difficulty_losing_weight=s.difficultyLosingWeight, fatigue=s.fatigue, mood_swings=s.moodSwings,
        pelvic_pain=s.pelvicPain, infertility=s.infertility, family_history=c.familyHistory,
        physical_activity=c.physicalActivity, stress_level=c.stressLevel, sleep_duration=c.sleepDuration,
        fsh=c.fsh, lh=c.lh, amh=c.amh, testosterone=c.testosterone, fasting_insulin=c.fastingInsulin,
        blood_glucose=c.bloodGlucose,
        risk_score=risk_score, risk_level=risk_level, confidence=confidence,
        recommendations=json.dumps(recs), top_features=json.dumps(contribs_sorted)
    )
    
    db.add(db_assessment)
    db.commit()
    db.refresh(db_assessment)
    
    return {
        "assessment_id": db_assessment.id,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "confidence": confidence,
        "top_features": contribs_sorted,
        "recommendations": recs,
        "clinical_details": {
            "bmi": round(bmi_val, 1),
            "lh_fsh_ratio": round(lh / fsh, 2) if (lh and fsh) else None
        }
    }

# --- User History & Trends ---
@app.get("/api/history")
def get_user_history(token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    assessments = db.query(Assessment).filter(Assessment.user_id == user.id).order_index(Assessment.created_at.desc()).all()
    
    history_list = []
    for a in assessments:
        history_list.append({
            "id": a.id,
            "risk_score": a.risk_score,
            "risk_level": a.risk_level,
            "created_at": a.created_at.strftime("%Y-%m-%d %H:%M"),
            "bmi": round(a.bmi, 1)
        })
    return history_list

@app.get("/api/trends")
def get_user_trends(token: str, db: Session = Depends(get_db)):
    user = get_current_user(token, db)
    # Get last 10 assessments ordered chronologically
    assessments = db.query(Assessment).filter(Assessment.user_id == user.id).order_by(Assessment.created_at.asc()).limit(10).all()
    
    trends = []
    for a in assessments:
        trends.append({
            "date": a.created_at.strftime("%b %d"),
            "score": a.risk_score
        })
    return trends

# --- Period History Persistence Endpoints ---
@app.post("/api/period-history")
def save_period_entry(entry: PeriodEntryCreate, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            user = get_current_user(token, db)
        except HTTPException:
            pass

    if not user:
        return {"status": "guest", "message": "Logged locally in browser. Sign in to sync across devices."}

    db_entry = PeriodEntry(
        user_id=user.id,
        start_date=entry.startDate,
        end_date=entry.endDate,
        notes=entry.notes,
        duration_days=entry.durationDays or 1
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return {"status": "success", "id": db_entry.id}

@app.get("/api/period-history")
def get_period_history_api(authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            user = get_current_user(token, db)
        except HTTPException:
            pass

    if not user:
        return []

    entries = db.query(PeriodEntry).filter(PeriodEntry.user_id == user.id).order_by(PeriodEntry.created_at.desc()).all()
    return [
        {
            "id": str(e.id),
            "startDate": e.start_date,
            "endDate": e.end_date,
            "durationDays": e.duration_days,
            "notes": e.notes
        }
        for e in entries
    ]


# --- Free-Text Health & Emotional Journal Endpoints ---

def extract_journal_insights(free_text: str, selected_symptoms: List[str]) -> Dict[str, Any]:
    text_lower = (free_text or "").lower()
    
    emotion = "neutral"
    if any(w in text_lower for w in ["scared", "fear", "terrified", "frightened"]):
        emotion = "fear"
    elif any(w in text_lower for w in ["anxious", "anxiety", "worried", "worry", "nervous", "stress", "stressed"]):
        emotion = "anxiety"
    elif any(w in text_lower for w in ["happy", "excited", "glad", "wonderful", "great", "relieved"]):
        emotion = "excitement"
    elif any(w in text_lower for w in ["frustrated", "exhausted", "tired of", "sick of", "annoyed"]):
        emotion = "frustration"
    elif any(w in text_lower for w in ["sad", "low", "depressed", "down", "upset"]):
        emotion = "sadness"
    elif any(w in text_lower for w in ["confused", "don't understand", "puzzled"]):
        emotion = "confusion"
        
    extracted_symptoms = list(selected_symptoms or [])
    if ("acne" in text_lower or "pimple" in text_lower) and "acne" not in [s.lower() for s in extracted_symptoms]:
        extracted_symptoms.append("Hormonal Acne / Skin Changes")
    if ("irregular" in text_lower or "delayed" in text_lower or "late period" in text_lower) and "irregular" not in [s.lower() for s in extracted_symptoms]:
        extracted_symptoms.append("Irregular Cycles")
    if ("bloat" in text_lower or "pain" in text_lower or "cramp" in text_lower) and "pain" not in [s.lower() for s in extracted_symptoms]:
        extracted_symptoms.append("Pelvic Discomfort")
    if ("tired" in text_lower or "fatigue" in text_lower or "exhausted" in text_lower) and "fatigue" not in [s.lower() for s in extracted_symptoms]:
        extracted_symptoms.append("Fatigue & Energy Slumps")

    return {
        "ai_label": "AI Structured Interpretation",
        "detected_emotion": emotion,
        "reported_symptoms": extracted_symptoms,
        "extracted_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M")
    }


@app.post("/api/journal")
def create_journal_entry(
    entry: JournalEntryCreate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            user = get_current_user(token, db)
        except HTTPException:
            pass

    if not user:
        raise HTTPException(status_code=401, detail="Please sign in to save your personal health journal.")

    entry_date = entry.entryDate or datetime.datetime.utcnow().strftime("%Y-%m-%d")
    insights = extract_journal_insights(entry.freeText or "", entry.selectedSymptoms or [])
    
    db_entry = HealthJournalEntry(
        user_id=user.id,
        entry_date=entry_date,
        free_text=entry.freeText or "",
        selected_symptoms=json.dumps(entry.selectedSymptoms or []),
        tags=json.dumps(entry.tags or []),
        emotional_context=insights.get("detected_emotion", "neutral"),
        extracted_insights=json.dumps(insights),
        consent_for_ai=entry.consentForAI if entry.consentForAI is not None else True
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    
    return {
        "status": "success",
        "entry": {
            "id": db_entry.id,
            "entryDate": db_entry.entry_date,
            "freeText": db_entry.free_text,
            "selectedSymptoms": json.loads(db_entry.selected_symptoms),
            "tags": json.loads(db_entry.tags),
            "emotionalContext": db_entry.emotional_context,
            "extractedInsights": json.loads(db_entry.extracted_insights),
            "consentForAI": db_entry.consent_for_ai,
            "createdAt": db_entry.created_at.strftime("%Y-%m-%d %H:%M")
        }
    }


@app.get("/api/journal")
def get_journal_entries(
    search: Optional[str] = None,
    startDate: Optional[str] = None,
    endDate: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            user = get_current_user(token, db)
        except HTTPException:
            pass

    if not user:
        return []

    query = db.query(HealthJournalEntry).filter(HealthJournalEntry.user_id == user.id)
    
    if search:
        query = query.filter(HealthJournalEntry.free_text.ilike(f"%{search}%"))
    if startDate:
        query = query.filter(HealthJournalEntry.entry_date >= startDate)
    if endDate:
        query = query.filter(HealthJournalEntry.entry_date <= endDate)
        
    entries = query.order_by(HealthJournalEntry.created_at.desc()).all()
    
    return [
        {
            "id": e.id,
            "entryDate": e.entry_date,
            "freeText": e.free_text,
            "selectedSymptoms": json.loads(e.selected_symptoms) if e.selected_symptoms else [],
            "tags": json.loads(e.tags) if e.tags else [],
            "emotionalContext": e.emotional_context,
            "extractedInsights": json.loads(e.extracted_insights) if e.extracted_insights else None,
            "consentForAI": e.consent_for_ai,
            "createdAt": e.created_at.strftime("%Y-%m-%d %H:%M")
        }
        for e in entries
    ]


@app.put("/api/journal/{entry_id}")
def update_journal_entry(
    entry_id: int,
    updates: JournalEntryUpdate,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            user = get_current_user(token, db)
        except HTTPException:
            pass

    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    entry = db.query(HealthJournalEntry).filter(
        HealthJournalEntry.id == entry_id,
        HealthJournalEntry.user_id == user.id
    ).first()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")

    if updates.freeText is not None:
        entry.free_text = updates.freeText
    if updates.selectedSymptoms is not None:
        entry.selected_symptoms = json.dumps(updates.selectedSymptoms)
    if updates.tags is not None:
        entry.tags = json.dumps(updates.tags)
    if updates.consentForAI is not None:
        entry.consent_for_ai = updates.consentForAI
        
    insights = extract_journal_insights(entry.free_text, json.loads(entry.selected_symptoms or "[]"))
    entry.emotional_context = insights.get("detected_emotion", "neutral")
    entry.extracted_insights = json.dumps(insights)
    
    db.commit()
    db.refresh(entry)
    
    return {
        "status": "success",
        "entry": {
            "id": entry.id,
            "entryDate": entry.entry_date,
            "freeText": entry.free_text,
            "selectedSymptoms": json.loads(entry.selected_symptoms),
            "tags": json.loads(entry.tags),
            "emotionalContext": entry.emotional_context,
            "extractedInsights": json.loads(entry.extracted_insights),
            "consentForAI": entry.consent_for_ai
        }
    }


@app.delete("/api/journal/{entry_id}")
def delete_journal_entry(
    entry_id: int,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            user = get_current_user(token, db)
        except HTTPException:
            pass

    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    entry = db.query(HealthJournalEntry).filter(
        HealthJournalEntry.id == entry_id,
        HealthJournalEntry.user_id == user.id
    ).first()
    
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")

    db.delete(entry)
    db.commit()
    return {"status": "success", "message": "Journal entry deleted successfully"}


@app.get("/api/journal/export")
def export_journal_data(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            user = get_current_user(token, db)
        except HTTPException:
            pass

    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    entries = db.query(HealthJournalEntry).filter(HealthJournalEntry.user_id == user.id).order_by(HealthJournalEntry.entry_date.desc()).all()
    
    return {
        "export_date": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M"),
        "user_email": user.email,
        "total_entries": len(entries),
        "journal_records": [
            {
                "id": e.id,
                "entry_date": e.entry_date,
                "free_text": e.free_text,
                "selected_symptoms": json.loads(e.selected_symptoms) if e.selected_symptoms else [],
                "tags": json.loads(e.tags) if e.tags else [],
                "emotional_context": e.emotional_context,
                "consent_for_ai": e.consent_for_ai,
                "created_at": e.created_at.strftime("%Y-%m-%d %H:%M")
            }
            for e in entries
        ]
    }


# --- OpenAI Conversational AI Endpoint ---
@app.post("/api/chat")
def chatbot_endpoint(req: ChatRequest, authorization: Optional[str] = Header(None), db: Session = Depends(get_db)):
    # Authenticate user if token provided
    current_user = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        try:
            current_user = get_current_user(token, db)
        except HTTPException:
            pass

    user_id = current_user.id if current_user else None

    # Load or create conversation session
    conv_id = req.conversation_id
    if not conv_id:
        conv_id = f"conv_{uuid.uuid4().hex[:12]}"
        if user_id:
            db_session = ConversationSession(id=conv_id, user_id=user_id, title="PCOS Health Consultation")
            db.add(db_session)
            db.commit()

    # Retrieve past message context from database if available
    db_history = []
    if user_id:
        messages = db.query(ChatMessage).filter(ChatMessage.conversation_id == conv_id).order_by(ChatMessage.created_at.asc()).all()
        for m in messages:
            db_history.append({"role": m.sender, "content": m.content})

    # Combine DB history with request payload history
    combined_history = db_history if db_history else req.history

    # Execute OpenAI conversation service loop
    ai_result = run_openai_chat(
        user_message=req.message,
        history_messages=combined_history,
        db=db,
        user_id=user_id
    )

    # Persist chat messages to database for logged in user
    if user_id:
        try:
            # Ensure conversation exists
            conv = db.query(ConversationSession).filter(ConversationSession.id == conv_id).first()
            if not conv:
                conv = ConversationSession(id=conv_id, user_id=user_id, title=req.message[:30])
                db.add(conv)
                db.commit()

            # Save user message
            user_msg = ChatMessage(conversation_id=conv_id, sender="user", content=req.message)
            db.add(user_msg)

            # Save assistant reply
            ai_msg = ChatMessage(conversation_id=conv_id, sender="assistant", content=ai_result["reply"])
            db.add(ai_msg)
            db.commit()
        except Exception as e:
            db.rollback()

    return {
        "reply": ai_result["reply"],
        "conversation_id": conv_id,
        "tool_calls_executed": ai_result.get("tool_calls_executed", [])
    }


# --- PDF Report Generation Endpoint ---
@app.get("/api/report/{assessment_id}")
def generate_pdf_report(assessment_id: int, db: Session = Depends(get_db)):
    assessment = db.query(Assessment).filter(Assessment.id == assessment_id).first()
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment report not found")
        
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    # Custom Styles matching the feminine soft theme
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#d946ef'), # Soft Purple/Pink
        spaceAfter=15
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#6b21a8'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#374151'),
        leading=14
    )
    
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12
    )

    elements = []
    
    # Title
    elements.append(Paragraph("PCOS/PCOD Health Assessment Report", title_style))
    elements.append(Paragraph(f"Generated on: {assessment.created_at.strftime('%Y-%m-%d %H:%M')}", body_style))
    elements.append(Paragraph("This report compiles assessment inputs, machine learning risk levels, and clinical recommendations.", body_style))
    elements.append(Spacer(1, 15))
    
    # Risk Score Summary Card
    risk_color = '#ef4444' if assessment.risk_level == "High" else ('#eab308' if assessment.risk_level == "Moderate" else '#10b981')
    risk_summary_data = [
        [Paragraph(f"<b>RISK LEVEL: {assessment.risk_level.upper()}</b>", ParagraphStyle('RiskText', parent=title_style, textColor=colors.HexColor(risk_color), fontSize=16)),
         Paragraph(f"<b>Risk Score: {int(assessment.risk_score)}%</b><br/>Confidence: {int(assessment.confidence * 100)}%", body_style)]
    ]
    summary_table = Table(risk_summary_data, colWidths=[250, 250])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#fdf2f8')), # Light pink bg
        ('PADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LINEBELOW', (0,0), (-1,-1), 2, colors.HexColor(risk_color)),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))
    
    # Clinical parameters
    elements.append(Paragraph("Patient Clinical Parameters", section_title_style))
    clinical_data = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Age", f"{assessment.age} years", "BMI", f"{round(assessment.bmi, 1)} kg/m²"],
        ["Height", f"{assessment.height} cm", "Weight", f"{assessment.weight} kg"],
        ["Cycle Length", f"{assessment.cycle_length} days", "Period Duration", f"{assessment.cycle_period} days"],
        ["FSH", f"{assessment.fsh} mIU/mL" if assessment.fsh else "—", "LH", f"{assessment.lh} mIU/mL" if assessment.lh else "—"],
        ["AMH", f"{assessment.amh} ng/mL" if assessment.amh else "—", "Testosterone", f"{assessment.testosterone} ng/dL" if assessment.testosterone else "—"],
        ["Fasting Insulin", f"{assessment.fasting_insulin} µIU/mL" if assessment.fasting_insulin else "—", "Blood Glucose", f"{assessment.blood_glucose} mg/dL" if assessment.blood_glucose else "—"],
        ["Stress Level", f"{assessment.stress_level}/10", "Sleep Duration", f"{assessment.sleep_duration} hours/night"],
        ["Physical Activity", "Low" if assessment.physical_activity == 1 else ("Moderate" if assessment.physical_activity == 2 else "High"), "Family History of PCOS", "Yes" if assessment.family_history else "No"]
    ]
    
    ct_table = Table(clinical_data, colWidths=[130, 120, 130, 120])
    ct_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f5f3ff')), # Lavender bg
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#6b21a8')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    elements.append(ct_table)
    elements.append(Spacer(1, 15))
    
    # Symptoms
    elements.append(Paragraph("Reported Symptoms", section_title_style))
    symptoms_list = []
    if assessment.irregular_periods: symptoms_list.append("Irregular periods")
    if assessment.no_periods: symptoms_list.append("Missed / absent periods")
    if assessment.excess_hair: symptoms_list.append("Excess facial/body hair (Hirsutism)")
    if assessment.acne: symptoms_list.append("Severe acne or oily skin")
    if assessment.hair_loss: symptoms_list.append("Hair thinning / loss")
    if assessment.dark_patches: symptoms_list.append("Dark patches of skin (Acanthosis nigricans)")
    if assessment.weight_gain: symptoms_list.append("Unexplained weight gain")
    if assessment.difficulty_losing_weight: symptoms_list.append("Difficulty losing weight")
    if assessment.fatigue: symptoms_list.append("Chronic fatigue")
    if assessment.mood_swings: symptoms_list.append("Mood swings / anxiety")
    if assessment.pelvic_pain: symptoms_list.append("Pelvic pain")
    if assessment.infertility: symptoms_list.append("Difficulty conceiving (Infertility)")
    
    if not symptoms_list:
        elements.append(Paragraph("No major symptoms reported.", body_style))
    else:
        symptoms_str = ", ".join(symptoms_list)
        elements.append(Paragraph(symptoms_str, body_style))
        
    elements.append(Spacer(1, 15))
    
    # Explanations / Contributing Factors
    elements.append(Paragraph("Explainable AI — Key Risk Drivers", section_title_style))
    elements.append(Paragraph("The machine learning model determined the following features contributed most to your risk score:", body_style))
    elements.append(Spacer(1, 5))
    
    top_feats = json.loads(assessment.top_features)
    feats_data = [["Feature", "Relative Impact", "Your Value"]]
    for f in top_feats:
        impact = "High Risk Contributor" if f['contribution'] > 0 else "Mitigating Factor"
        feats_data.append([f['feature'], impact, str(f['value'])])
        
    feats_table = Table(feats_data, colWidths=[150, 200, 150])
    feats_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f5f3ff')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('PADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e5e7eb')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    elements.append(feats_table)
    elements.append(Spacer(1, 15))
    
    # Recommendations
    elements.append(Paragraph("Personalized Lifestyle & Clinical Recommendations", section_title_style))
    recs = json.loads(assessment.recommendations)
    for r in recs:
        elements.append(Paragraph(f"• {r}", body_style))
        elements.append(Spacer(1, 3))
        
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("<i>Disclaimer: This report is generated by a machine learning model for informational screening. It is not a clinical diagnosis. Please consult a qualified gynecologist to discuss your symptoms and get an ultrasound confirmation.</i>", ParagraphStyle('Disclaimer', parent=body_style, fontSize=8, textColor=colors.HexColor('#9ca3af'))))
    
    doc.build(elements)
    buffer.seek(0)
    
    # Return as Streaming PDF file
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=PCOS_Report_{assessment_id}.pdf"})

# --- Automatic Retraining Background Task ---
def retrain_model_task():
    try:
        db = SessionLocal()
        assessments = db.query(Assessment).all()
        if len(assessments) < 10: # Retrain only when we have sufficient new data
            db.close()
            return
            
        print("Starting automatic model retraining background task...")
        
        # Load synthetic baseline data
        csv_path = os.path.join(BASE_DIR, 'pcod_dataset.csv')
        baseline_df = pd.read_csv(csv_path)
        
        # Convert new user assessments to training format
        new_records = []
        for a in assessments:
            # We assume label is 1 if score > 50, else 0 (self-learning/pseudo-labeling feedback loop)
            # or if the user clicks a feedback confirmation button (which we can implement).
            label = 1 if a.risk_score >= 50 else 0
            
            # Map database parameters to feature dict
            cycle_reg = a.cycle_regularity
            
            new_records.append({
                'Age': a.age, 'Height': a.height, 'Weight': a.weight, 'BMI': a.bmi,
                'CycleRegularity': cycle_reg, 'CycleLength': a.cycle_length, 'CyclePeriod': a.cycle_period,
                'Hirsutism': int(a.excess_hair), 'Acne': int(a.acne), 'HairLoss': int(a.hair_loss),
                'DarkPatches': int(a.dark_patches), 'WeightGain': int(a.weight_gain),
                'DifficultyLosingWeight': int(a.difficulty_losing_weight), 'Fatigue': int(a.fatigue),
                'MoodSwings': int(a.mood_swings), 'PelvicPain': int(a.pelvic_pain), 'Infertility': int(a.infertility),
                'FamilyHistory': int(a.family_history), 'PhysicalActivity': a.physical_activity,
                'StressLevel': a.stress_level, 'SleepDuration': a.sleep_duration,
                'FSH': a.fsh if a.fsh else 5.5, 'LH': a.lh if a.lh else 6.0,
                'AMH': a.amh if a.amh else 2.5, 'Testosterone': a.testosterone if a.testosterone else 35.0,
                'FastingInsulin': a.fasting_insulin if a.fasting_insulin else 8.0,
                'BloodGlucose': a.blood_glucose if a.blood_glucose else 90.0,
                'PCOS': label
            })
            
        new_df = pd.DataFrame(new_records)
        combined_df = pd.concat([baseline_df, new_df], ignore_index=True)
        
        # Save combined dataset back
        combined_df.to_csv(csv_path, index=False)
        
        # Trigger retraining
        from backend.train_models import load_and_preprocess_data, train_and_evaluate
        X, y = load_and_preprocess_data(csv_path)
        train_and_evaluate(X, y)
        print("Automatic model retraining completed successfully!")
        
        db.close()
    except Exception as e:
        print(f"Error during model retraining: {e}")

@app.post("/api/retrain")
def trigger_retraining(background_tasks: BackgroundTasks):
    background_tasks.add_task(retrain_model_task)
    return {"message": "Model retraining task queued in the background."}
