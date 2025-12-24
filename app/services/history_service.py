from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.history import AiChatbot, AiType
from datetime import date
from typing import List, Optional

class HistoryService:
    async def save_chat_history(
        self,
        db: AsyncSession,
        user_id: int,
        ai_type: str,
        question: str,
        answer: str,
        ref_date: Optional[date] = None
    ):
        """
        대화 내용을 DB에 저장합니다.
        """
        try:
            # AiType Enum String Check or Default
            try:
                # MM-BE defines FeedBack, Recommendation but here we use generic types if needed
                # ai_type passed from endpoint (e.g., 'CHAT', 'FEEDBACK')
                valid_type = ai_type
            except:
                valid_type = "CHAT"

            new_entry = AiChatbot(
                user_id=user_id,
                ai_type=valid_type, # String column
                user_question=question,
                ai_response=answer,
                ref_date=ref_date
            )
            db.add(new_entry)
            await db.commit()
            await db.refresh(new_entry)
            print(f"✅ History Saved: ID={new_entry.id}, Type={ai_type}")
            return new_entry
        except Exception as e:
            print(f"❌ Failed to save history: {e}")
            await db.rollback()
            return None

    async def get_chat_history(self, db: AsyncSession, user_id: int, limit: int = 50) -> List[dict]:
        """
        사용자의 최근 대화 내역을 조회합니다.
        """
        try:
            # 최근 순으로 조회
            stmt = select(AiChatbot).where(AiChatbot.user_id == user_id).order_by(AiChatbot.id.desc()).limit(limit)
            result = await db.execute(stmt)
            rows = result.scalars().all()
            
            # Entity -> Dict -> Frontend Format
            # Frontend expects: {id, type, text (answer), question, createdAt}
            formatted = []
            for row in rows:
                formatted.append(row.to_dict())
            
            return formatted
        except Exception as e:
            print(f"❌ Failed to fetch history: {e}")
            return []

history_service = HistoryService()
