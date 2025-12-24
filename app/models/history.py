from sqlalchemy import Column, Integer, String, Text, Date, DateTime, Enum, BigInteger
from sqlalchemy.sql import func
from app.core.database import Base
import enum

class AiType(enum.Enum):
    FEEDBACK = "FEEDBACK"
    RECOMMENDATION = "RECOMMENDATION"
    CHAT = "CHAT"
    MEAL_PLAN = "MEAL_PLAN" # Added for Meal Plan API

class AiChatbot(Base):
    __tablename__ = "ai_chatbot"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)
    ref_date = Column(Date, nullable=True) # 기준 날짜 (피드백 대상 등)
    ai_type = Column(String(20), nullable=False) # Enum as String
    user_question = Column(Text, nullable=True)
    ai_response = Column(Text, nullable=True)
    
    # BaseEntity fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.ai_type,
            "question": self.user_question,
            "answer": self.ai_response,
            "createdAt": self.created_at.isoformat() if self.created_at else None
        }
