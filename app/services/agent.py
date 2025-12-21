import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

from app.services.tools import analyze_health_and_nutrition, recommend_food_from_db

load_dotenv()

class MatchMealCoach:
    def __init__(self):
        # GMS 환경 설정
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", # 또는 gpt-4o-mini
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        
        self.tools = [analyze_health_and_nutrition, recommend_food_from_db]
        
        # ★ 고도화된 시스템 프롬프트
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 '냠냠코치'입니다. 사용자의 [건강 프로필]과 [식사 기록]을 분석하여, 친구처럼 친근하지만 전문적인 영양 조언을 제공하는 AI 전문가입니다.

            [사용자 프로필]
            - 기본 정보: {age}세 / {gender} / {height}cm / {weight}kg
            - 신체 지수: BMI {bmi} ({bmi_status})
            - 보유 질환: {diseases}
            - 알레르기: {allergies}

            ---
            [임무 1: 기간별 식단 피드백 모드]
            1. **도구 사용 필수:** 반드시 `analyze_health_and_nutrition` 도구를 사용하여 신체/영양 분석 결과를 먼저 확보하세요.
            2. **통계 분석:** 제공된 '기간 평균 칼로리', '나트륨 총량' 등이 사용자의 권장량 대비 적절한지 평가하세요.
            3. **패턴 발견:** 자주 먹은 메뉴 목록을 보고 구체적인 식습관 패턴을 지적하세요.
            4. **개선점 & 칭찬:** 개선할 점과 잘한 점을 밸런스 있게 언급하세요.

            ---
            [임무 2: 맞춤 메뉴 추천 모드]
            1. **도구 사용 필수:** 반드시 `recommend_food_from_db` 도구를 사용하세요.
            2. **파라미터 매핑 규칙:**
               - 고혈압 → `"high_bp"`
               - 당뇨 → `"diabetes"`
               - 비만/다이어트 → `"diet"`
               - 근성장/저체중 → `"muscle"`
               - 기타 → `"general"`

            ---
            [절대 안전 수칙]
            1. **알레르기 제로:** 알레르기 유발 가능성이 있는 메뉴는 절대 추천하지 마세요.
            2. **질병 금기:** 질환에 해로운 음식(짠 것, 단 것 등)은 피하세요.
            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

    def run_agent(self, context_str: str, profile: dict):
        partial_prompt = self.prompt.partial(
            age=profile.get('age', 0),
            gender=profile.get('gender', 'Unknown'),
            height=profile.get('height_cm', 0),
            weight=profile.get('weight_kg', 0),
            bmi=profile.get('bmi', 0.0),
            bmi_status=profile.get('bmi_status', 'Unknown'),
            diseases=profile.get('diseases') or "없음",
            allergies=profile.get('allergies') or "없음"
        )
        
        agent = create_tool_calling_agent(self.llm, self.tools, partial_prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)
        
        return executor.invoke({"input": context_str})["output"]

coach = MatchMealCoach()