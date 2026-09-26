from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import datetime
import os

db_url_env = os.getenv("DATABASE_URL", "")
if db_url_env:
    if db_url_env.startswith("postgres://"):
        db_url_env = db_url_env.replace("postgres://", "postgresql://", 1)
    DATABASE_URL = db_url_env
else:
    if os.getenv("VERCEL"):
        DATABASE_URL = "sqlite:///" + os.path.join(os.environ.get("TMPDIR", "/tmp"), "pcod_system.db")
    else:
        DATABASE_URL = "sqlite:///backend/pcod_system.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    assessments = relationship("Assessment", back_populates="user", cascade="all, delete-orphan")
    period_entries = relationship("PeriodEntry", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("ConversationSession", back_populates="user", cascade="all, delete-orphan")
    journal_entries = relationship("HealthJournalEntry", back_populates="user", cascade="all, delete-orphan")

class Assessment(Base):
    __tablename__ = "assessments"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    # Inputs
    age = Column(Integer)
    height = Column(Float)
    weight = Column(Float)
    bmi = Column(Float)
    cycle_regularity = Column(Integer)  # 0 or 1
    cycle_length = Column(Float)
    cycle_period = Column(Integer)
    
    # Symptoms
    irregular_periods = Column(Boolean, default=False)
    no_periods = Column(Boolean, default=False)
    excess_hair = Column(Boolean, default=False)
    acne = Column(Boolean, default=False)
    hair_loss = Column(Boolean, default=False)
    dark_patches = Column(Boolean, default=False)
    weight_gain = Column(Boolean, default=False)
    difficulty_losing_weight = Column(Boolean, default=False)
    fatigue = Column(Boolean, default=False)
    mood_swings = Column(Boolean, default=False)
    pelvic_pain = Column(Boolean, default=False)
    infertility = Column(Boolean, default=False)
    
    # Lifestyle & Medical
    family_history = Column(Boolean, default=False)
    physical_activity = Column(Integer)
    stress_level = Column(Integer)
    sleep_duration = Column(Float)
    fsh = Column(Float, nullable=True)
    lh = Column(Float, nullable=True)
    amh = Column(Float, nullable=True)
    testosterone = Column(Float, nullable=True)
    fasting_insulin = Column(Float, nullable=True)
    blood_glucose = Column(Float, nullable=True)
    
    # Predictions
    risk_score = Column(Float)  # 0 to 100 percentage
    risk_level = Column(String)  # "Low", "Moderate", "High"
    confidence = Column(Float)  # 0.0 to 1.0
    recommendations = Column(Text)  # JSON-encoded array
    top_features = Column(Text)  # JSON-encoded array (XAI)
    
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="assessments")

class PeriodEntry(Base):
    __tablename__ = "period_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    start_date = Column(String, nullable=False)
    end_date = Column(String, nullable=False)
    duration_days = Column(Integer, default=1)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="period_entries")

class HealthJournalEntry(Base):
    __tablename__ = "health_journal_entries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    entry_date = Column(String, nullable=False, index=True) # YYYY-MM-DD
    free_text = Column(Text, nullable=True)
    selected_symptoms = Column(Text, nullable=True) # JSON string array
    tags = Column(Text, nullable=True) # JSON string array
    emotional_context = Column(String, nullable=True) # e.g. "anxious", "excited", "worried", "neutral"
    extracted_insights = Column(Text, nullable=True) # JSON string object
    consent_for_ai = Column(Boolean, default=True) # User privacy consent
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="journal_entries")

class ConversationSession(Base):
    __tablename__ = "conversations"
    
    id = Column(String, primary_key=True, index=True) # UUID string
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, default="PCOS Health Consultation")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    user = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    sender = Column(String, nullable=False) # 'user', 'assistant', 'system', 'tool'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    conversation = relationship("ConversationSession", back_populates="messages")

class Hospital(Base):
    __tablename__ = "hospitals"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    state = Column(String, index=True)
    district = Column(String, index=True)
    address = Column(String)
    phone = Column(String, nullable=True)
    specialty = Column(String, default="Gynecology & Obstetrics")

def init_db():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    
    # Seed mock Indian hospitals if the table is empty
    if db.query(Hospital).count() == 0:
        hospitals_seed = [
            # Karnataka
            {"name": "Manipal Hospital", "state": "Karnataka", "district": "Bengaluru", "address": "98, HAL Old Airport Rd, Kodihalli, Bengaluru", "phone": "+91-80-2502-4444"},
            {"name": "Cloudnine Hospital", "state": "Karnataka", "district": "Bengaluru", "address": "1533, 9th Main Rd, 3rd Block, Jayanagar, Bengaluru", "phone": "+91-80-6799-6000"},
            {"name": "Fortis Hospital", "state": "Karnataka", "district": "Bengaluru", "address": "154/9, Bannerghatta Road, Opposite IIM-B, Bengaluru", "phone": "+91-80-6621-4444"},
            {"name": "KMC Hospital", "state": "Karnataka", "district": "Mangaluru", "address": "Light House Hill Rd, Hampankatta, Mangaluru", "phone": "+91-824-244-5858"},
            
            # Maharashtra
            {"name": "Jaslok Hospital", "state": "Maharashtra", "district": "Mumbai", "address": "15, Dr. Deshmukh Marg, Pedder Rd, Mumbai", "phone": "+91-22-6657-3333"},
            {"name": "Kokilaben Dhirubhai Ambani Hospital", "state": "Maharashtra", "district": "Mumbai", "address": "Rao Saheb, Achutrao Patwardhan Marg, Four Bungalows, Andheri West, Mumbai", "phone": "+91-22-4269-6969"},
            {"name": "Sahyadri Super Speciality Hospital", "state": "Maharashtra", "district": "Pune", "address": "Plot No. 30C, Karve Rd, Deccan Gymkhana, Pune", "phone": "+91-20-6721-3000"},
            
            # Delhi
            {"name": "Max Super Speciality Hospital", "state": "Delhi", "district": "New Delhi", "address": "1-2, Press Enclave Road, Saket, New Delhi", "phone": "+91-11-2651-5050"},
            {"name": "Apollo Cradle & Children's Hospital", "state": "Delhi", "district": "New Delhi", "address": "R-2, Greater Kailash Part 1, New Delhi", "phone": "+91-11-4424-4424"},
            
            # Tamil Nadu
            {"name": "Apollo Women's Hospital", "state": "Tamil Nadu", "district": "Chennai", "address": "15, Shafee Mohammed Rd, Thousand Lights West, Chennai", "phone": "+91-44-2829-0200"},
            {"name": "Kauvery Hospital", "state": "Tamil Nadu", "district": "Chennai", "address": "199, Luz Church Rd, Mylapore, Chennai", "phone": "+91-44-4000-6000"},
            
            # Telangana
            {"name": "Rainbow Children's Hospital & BirthRight", "state": "Telangana", "district": "Hyderabad", "address": "Road No. 2, Banjara Hills, Near L V Prasad Eye Hospital, Hyderabad", "phone": "+91-40-4466-5555"}
        ]
        
        for h in hospitals_seed:
            db.add(Hospital(**h))
        db.commit()
        print("Mock Indian hospitals database seeded successfully.")
    
    db.close()

if __name__ == "__main__":
    init_db()
