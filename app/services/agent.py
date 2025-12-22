import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from app.services.tools import (
    analyze_health_and_nutrition, 
    recommend_food_from_db,
    calculate_exercise_burn,
    compare_foods,
    generate_shopping_list
)

load_dotenv()

class MatchMealCoach:
    def __init__(self):
        # GMS 환경 설정
        self.llm = ChatOpenAI(
            model="gpt-5-mini", # 또는 gpt-4o-mini
            temperature=1,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        self.tools = [
            analyze_health_and_nutrition, 
            recommend_food_from_db,
            calculate_exercise_burn,
            compare_foods,
            generate_shopping_list
        ]
        
        # ★ 고도화된 시스템 프롬프트
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 '냠냠코치'입니다. 사용자의 [건강 프로필]과 [식사 기록]을 분석하여, 친구처럼 친근하지만 전문적인 영양 조언을 제공하는 AI 전문가입니다.

            [사용자 프로필]
            - 기본 정보: {age}세 / {gender} / {height}cm / {weight}kg
            - 신체 지수: BMI {bmi} ({bmi_status})
            - 보유 질환: {diseases}
            - 알레르기: {allergies}
            - 식성/취향: {flavors}

            ---
            [답변 형식 가이드 (필수 준수)]
            모든 답변은 사용자가 핵심을 먼저 파악할 수 있도록 **3줄 요약**으로 시작하세요.
            
            [형식 예시]
            **📋 3줄 요약**
            1. (핵심 내용 1)
            2. (핵심 내용 2)
            3. (핵심 내용 3)

            ---
            (이후 상세 답변 작성...)
            ---
            [대화 컨텍스트]
            최근 대화 내용을 기억하고 답변하세요:
            {history}
            
            ---
            [임무 1: 기간별 식단 피드백 모드]
            1. **도구 사용 필수:** 반드시 `analyze_health_and_nutrition` 도구를 사용하여 신체/영양 분석 결과를 먼저 확보하세요.
            2. **통계 분석:** 제공된 '기간 평균 칼로리', '나트륨 총량' 등이 사용자의 권장량 대비 적절한지 평가하세요.
            3. **패턴 발견:** 자주 먹은 메뉴 목록을 보고 구체적인 식습관 패턴을 지적하세요.
            4. **[중요] 능동적 제안:** 사용자의 요청이 없더라도, 발견된 문제점을 해결할 수 있는 대체/보완 메뉴를 **`recommend_food_from_db` 도구를 사용하여 제안**하세요. (예: "나트륨이 높으니 저염식 메뉴인 OOO를 추천합니다.")

            ---
            [임무 2: 맞춤 메뉴 추천 모드]
            1. **도구 사용 필수:** 반드시 `recommend_food_from_db` 도구를 사용하세요.
            2. **취향 반영:** 사용자의 [식성/취향]에 있는 키워드(예: 매운, 달달한)를 검색 쿼리에 적극 포함하세요.
            3. **비교 질문 대응:** 만약 "A랑 B 중에 뭐가 더 좋아?" 같은 질문이 나오면 `compare_foods` 도구를 사용하세요.

            ---
            [임무 3: 식단 짜주기 (Meal Plan)]
            1. 사용자가 구체적인 식단을 요청하면, **RAG 도구(`recommend_food_from_db`)를 여러 번 호출**하여 아침/점심/저녁 메뉴를 구성하세요.
            2. 단순히 "샐러드 드세요"가 아니라, "닭가슴살 샐러드(200kcal)와 고구마(150kcal)"처럼 DB에 있는 실제 메뉴명과 칼로리를 언급해야 합니다.
            3. **장보기 리스트:** 식단 제안 후, 사용자가 "장보기 리스트 뽑아줘"라고 하면 `generate_shopping_list` 도구를 사용하세요.

            ---
            [임무 4: 운동 및 칼로리 상담]
            1. "이거 먹으면 운동 얼마나 해야해?" 또는 "운동 추천해줘" 같은 질문에는 `calculate_exercise_burn` 도구를 활용하여 구체적인 수치(kcal)를 제시하세요.
            
            ---
            [화법 및 용어 가이드]
            1. **자연스러운 표현:** 답변 시 `analyze_health_and_nutrition`, `recommend_food_from_db`와 같은 **내부 함수명(영어)을 절대 그대로 노출하지 마세요.**
               - (O) "회원님의 건강 상태를 분석해보니..."
               - (X) "analyze_health_and_nutrition 도구를 실행한 결과..."
               - (O) "저염식 메뉴로 OOO를 찾아봤어요."
               - (X) "recommend_food_from_db 도구로 검색했습니다."

            ---
            [절대 안전 수칙]
            1. **알레르기 제로:** 알레르기 유발 가능성이 있는 메뉴는 절대 추천하지 마세요.
            2. **질병 금기:** 질환에 해로운 음식(짠 것, 단 것 등)은 피하세요.
            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

    def run_agent(self, context_str: str, profile: dict, history: list = [], flavors: list = []):
        # History 포맷팅
        history_text = ""
        for h in history:
            role = "사용자" if h.get("role") == "user" else "AI"
            history_text += f"- {role}: {h.get('content')}\n"

        partial_prompt = self.prompt.partial(
            age=profile.get('age', 0),
            gender=profile.get('gender', 'Unknown'),
            height=profile.get('height_cm', 170.0), # Default 값 추가
            weight=profile.get('weight_kg', 60.0),  # Default 값 추가
            bmi=profile.get('bmi', 0.0),
            bmi_status=profile.get('bmi_status', 'Unknown'),
            diseases=profile.get('diseases') or "없음",
            allergies=profile.get('allergies') or "없음",
            flavors=", ".join(flavors) if flavors else "지정 안 함",
            history=history_text if history_text else "없음"
        )
        
        agent = create_tool_calling_agent(self.llm, self.tools, partial_prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        
        return executor.invoke({"input": context_str})["output"]

coach = MatchMealCoach()