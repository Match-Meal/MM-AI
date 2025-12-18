import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 벡터 DB 저장 경로
PERSIST_DIRECTORY = "./chroma_db"

class FoodVectorStore:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings()
        self.db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embedding_function,
            collection_name="food_collection"
        )

    # 음식 데이터 적재 (최초 1회 또는 배치 작업용)
    def add_foods(self, food_list: list):
        documents = []
        for food in food_list:
            # RAG 검색이 잘 되도록 텍스트화 (Metadata에는 실제 수치 저장)
            content = f"음식명: {food['name']}, 카테고리: {food['category']}, 특징: {food['desc']}"
            meta = {
                "name": food['name'],
                "calories": food['calories'],
                "protein": food['protein'],
                "fat": food['fat'],
                "carbs": food['carbohydrate']
            }
            documents.append(Document(page_content=content, metadata=meta))
        
        self.db.add_documents(documents)

    # 검색 (Agent가 사용할 함수)
    def search_food(self, query: str, k=3):
        return self.db.similarity_search(query, k=k)

food_store = FoodVectorStore() 