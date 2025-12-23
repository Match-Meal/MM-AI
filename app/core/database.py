import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

# 1. Environment Variables
RDS_USERNAME = os.getenv("RDS_USERNAME")
RDS_PASSWORD = os.getenv("RDS_PASSWORD")
RDS_HOST = os.getenv("RDS_HOST", "matchmeal-project.chmge8kcwil3.ap-northeast-2.rds.amazonaws.com")
RDS_DB_NAME = os.getenv("RDS_DB_NAME", "mm_db")

# 2. Database URL (Async MySQL)
# Format: mysql+aiomysql://user:password@host:port/dbname?charset=utf8mb4
DATABASE_URL = f"mysql+aiomysql://{RDS_USERNAME}:{RDS_PASSWORD}@{RDS_HOST}:3306/{RDS_DB_NAME}?charset=utf8mb4"

# 3. SQLAlchemy Async Engine
engine = create_async_engine(
    DATABASE_URL,
    echo=True, # Set to False in production
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# 4. Session Factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# 5. Declarative Base
Base = declarative_base()

# 6. Dependency Injection for FastAPI
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
