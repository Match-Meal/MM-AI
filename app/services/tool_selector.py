from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from app.services.vector_store import tool_store
import json
import os
from dotenv import load_dotenv

load_dotenv()

class ToolSelector:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 주어진 질문을 해결하기 위해 가장 적절한 도구를 선택하는 AI 어시스턴트입니다.
            
            [사용 가능한 도구 목록]
            {candidate_tools}
            
            [지시사항]
            1. 사용자의 질문을 분석하여 위 도구 목록 중 하나 이상이 필요한지 판단하세요.
            2. 도구가 필요하다면 해당 도구의 정확한 이름을 리스트로 반환하세요.
            3. 도구가 전혀 필요 없는 단순 대화(인사, 날씨, 농담 등)라면 빈 리스트 []를 반환하세요.
            4. 반드시 아래 JSON 형식으로만 응답하세요.
            
            {{
                "reasoning": "선택 이유",
                "selected_tools": ["tool_name1", ...]
            }}
            """),
            ("human", "{question}")
        ])

    def select_tools(self, query: str, tools_map: dict) -> list[str]:
        """
        사용자 쿼리에 적합한 도구 이름을 반환합니다.
        Step 1: Vector DB에서 후보 도구 검색
        Step 2: LLM이 최종 선별
        """
        # 1. Vector Search for Candidates (Recall)
        # 도구 개수가 매우 적으므로(약 14개), 벡터 검색보다는 
        # 그냥 모든 도구를 후보로 LLM에게 전달하는 것이 더 정확하고 안정적입니다.
        # (Chroma HNSW의 "ef or M is too small" 에러 방지 목적 포함)
        try:
            # 전체 도구 리스트 가져오기
            candidates = tool_store.all_tools_docs()
            
            # 만약 도구가 너무 많아지면(예: 20개 이상) 그때만 벡터 검색 수행
            if len(candidates) > 20:
                candidates = tool_store.search_tools(query, k=10)
        except Exception as e:
            print(f"Tool Selection Error: {e}")
            # Fallback: 검색 실패 시 빈 리스트
            return []

        if not candidates:
            return []
            
        # 후보군 텍스트 생성
        candidates_text = ""
        valid_tool_names = set(tools_map.keys())
        
        filtered_candidates = []
        for doc in candidates:
            name = doc.metadata.get('name')
            if name in valid_tool_names:
                filtered_candidates.append(f"- {name}: {doc.page_content}")
        
        if not filtered_candidates:
            return []
            
        candidates_str = "\n".join(filtered_candidates)
        
        # 2. LLM Select (Precision)
        chain = self.prompt | self.llm
        try:
            res = chain.invoke({
                "candidate_tools": candidates_str,
                "question": query
            })
            
            # JSON Parsing
            content = res.content
            # Markdown code block 제거
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            data = json.loads(content)
            selected = data.get("selected_tools", [])
            
            # 유효성 검증
            final_tools = [name for name in selected if name in valid_tool_names]
            
            print(f"🧐 Query: {query}")
            print(f"   Candidates: {[doc.metadata.get('name') for doc in candidates]}")
            print(f"   Selected: {final_tools}")
            
            return final_tools
            
        except Exception as e:
            print(f"Tool Selection LLM Error: {e}")
            return []

tool_selector = ToolSelector()
