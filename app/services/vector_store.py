import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIRECTORY = "./chroma_db"

class FoodVectorStore:
    def __init__(self):
        # ★ GMS 환경 설정 적용
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        self.db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embedding_function,
            collection_name="food_collection"
        )

    # ★ CSV 파일 로드 및 적재
    def load_from_csvs(self):
        # 이미 데이터가 있는지 확인 (중복 적재 방지)
        try:
            if self.db._collection.count() > 0:
                print(f"✅ ChromaDB에 이미 {self.db._collection.count()}개의 데이터가 있습니다. 초기화를 건너뜁니다.")
                return
        except Exception as e:
            print(f"⚠️ DB 확인 중 오류 (무시): {e}")

        import csv
        documents = []
        
        # 파일별 매핑 설정 [파일경로, 헤더행인덱스, 인코딩, {필드명: 인덱스}]
        files_config = [
            {
                "path": "app/400_Food_DB.csv", "header_row": 0, "encoding": "utf-8",
                "map": {"name": 0, "kcal": 2, "carb": 3, "sugar": 4, "fat": 5, "prot": 6, "sodium": 9},
                "desc": "일반 음식"
            },
            {
                "path": "app/50000_Food_DB.csv", "header_row": 3, "encoding": "utf-8", # utf-8로 읽히는지 재확인 필요하지만 get_columns 성공했으므로 utf-8
                "map": {"name": 5, "kcal": 15, "carb": 21, "sugar": 22, "fat": 20, "prot": 19, "sodium": 45},
                "desc": "가공 식품"
            }
        ]

        print("🔄 CSV 데이터 적재 시작...")
        
        for config in files_config:
            fpath = config["path"]
            if not os.path.exists(fpath):
                print(f"⚠️ 파일 없음: {fpath}")
                continue
                
            try:
                # 50k DB가 utf-8로 성공했는지 확인 필요. Step 764 결과는 utf-8로 성공함.
                with open(fpath, 'r', encoding=config["encoding"]) as csvfile:
                    reader = csv.reader(csvfile)
                    # 헤더 건너뛰기
                    for _ in range(config["header_row"] + 1):
                        next(reader)
                    
                    for row in reader:
                        try:
                            # 인덱스 접근 안전 장치
                            m = config["map"]
                            if len(row) <= max(m.values()): continue
                            
                            name = row[m["name"]].strip()
                            if not name: continue
                            
                            def safe_float(val):
                                try: return float(val.replace(',', ''))
                                except: return 0.0

                            meta = {
                                "name": name,
                                "calories": safe_float(row[m["kcal"]]),
                                "carbohydrate": safe_float(row[m["carb"]]),
                                "sugar": safe_float(row[m["sugar"]]),
                                "fat": safe_float(row[m["fat"]]),
                                "protein": safe_float(row[m["prot"]]),
                                "sodium": safe_float(row[m["sodium"]])
                            }
                            
                            # 검색용 텍스트 생성
                            content = (f"음식명: {name}, 칼로리: {meta['calories']}kcal, "
                                       f"탄수: {meta['carbohydrate']}g, 단백: {meta['protein']}g, "
                                       f"지방: {meta['fat']}g, 당류: {meta['sugar']}g")
                                       
                            documents.append(Document(page_content=content, metadata=meta))
                            
                        except Exception as e:
                            continue # 개별 행 오류 무시
                            
            except Exception as e:
                print(f"❌ {fpath} 로드 실패: {e}")

        if documents:
            # 배치 단위로 추가 (너무 많으면 에러 가능성)
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                self.db.add_documents(batch)
                print(f"   -> {i+len(batch)} / {len(documents)} 저장 완료")
            print("✅ 모든 데이터 적재 완료!")
        else:
            print("⚠️ 적재할 데이터가 없습니다.")

    # 데이터 적재 (수동 추가용)
    def add_foods(self, food_list: list):
        documents = []
        for food in food_list:
            content = f"음식명: {food['name']}, 카테고리: {food.get('category','')}, 특징: {food.get('desc','')}"
            meta = {
                "name": food['name'],
                "calories": float(food['calories']),
                "protein": float(food['protein']),
                "fat": float(food['fat']),
                "carbohydrate": float(food['carbohydrate']),
                "sodium": float(food.get('sodium', 0)),
                "sugar": float(food.get('sugar', 0))
            }
            documents.append(Document(page_content=content, metadata=meta))
        
        if documents:
            self.db.add_documents(documents)

    # ★ 검색 (필터 기능 포함)
    def search_food(self, query: str, k=5, filter=None):
        try:
            # 데이터가 없는 경우 검색 생략 (에러 방지)
            if self.db._collection.count() == 0: return []
            
            if filter:
                return self.db.similarity_search(query, k=k, filter=filter)
            return self.db.similarity_search(query, k=k)
        except Exception as e:
            print(f"Food Search Error: {e}")
            return []

class ToolVectorStore:
    def __init__(self):
        self.embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        self.db = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embedding_function,
            collection_name="tool_collection"
        )

    def index_tools(self, tools: list):
        """LangChain 도구 리스트를 받아 벡터 DB에 저장합니다."""
        # 기존 데이터 확인 (간단하게 이름으로 중복 체크하거나, 매번 덮어쓰기)
        # 여기서는 매번 초기화 후 다시 저장하는 방식이 안전 (도구 설명 변경 반영)
        
        # 컬렉션 초기화가 까다로우므로, 간단히 모든 도구를 가져와서 이름 비교?
        # 또는 그냥 중복 각오하고 업데이트?
        # Chroma의 add_documents는 ID를 지정하면 업데이트가 됨.
        
        documents = []
        for tool in tools:
            # 도구 이름과 설명을 저장
            content = f"도구 이름: {tool.name}\n설명: {tool.description}"
            meta = {"name": tool.name}
            # ID는 도구 이름으로 고정하여 중복 적재 방지/업데이트
            documents.append(Document(page_content=content, metadata=meta, id=tool.name))
            
        if documents:
            # IDs list
            ids = [doc.id for doc in documents]
            # 이미 존재하는지 확인하지 않고 upsert(add는 id 있으면 에러날 수 있음, Chroma 최신은 upsert 지원 확인 필요)
            # Langchain Chroma wrapper: add_documents usually adds. 
            # safe approach: delete and add, or use specific update method.
            # let's try add_documents with ids. If langchain chroma doesn't support upsert by default, we might get dupes if ids not used.
            # Actually, Langchain Chroma `add_documents` usually generates distinct IDs if not provided.
            # If we provide IDs, it might error if exists.
            
            # Resetting collection for tools is safer as tools are few.
            try:
                # This is a bit hacky, but effective for small toolsets
                existing_ids = self.db.get()['ids']
                if existing_ids:
                    self.db.delete(ids=existing_ids)
            except:
                pass
                
            self.db.add_documents(documents)
            print(f"✅ {len(documents)}개의 도구가 인덱싱되었습니다.")

    def search_tools(self, query: str, k=3):
        return self.db.similarity_search(query, k=k)

    def all_tools_docs(self):
        """저장된 모든 도구 문서를 반환합니다."""
        res = self.db.get()
        docs = []
        if not res or not res['ids']: return []
        for i in range(len(res['ids'])):
            docs.append(Document(
                page_content=res['documents'][i],
                metadata=res['metadatas'][i]
            ))
        return docs

food_store = FoodVectorStore()
tool_store = ToolVectorStore()
