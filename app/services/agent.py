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
    generate_shopping_list,
    recommend_seasonal_food,
    recommend_food_for_symptom,
    get_recipe_procedure,
    check_food_compatibility,
    calculate_maintenance_calories,
    suggest_healthy_alternative,
    calculate_water_needs,
    recommend_snack,
    analyze_nutrient_deficiency
)
from app.services.tool_selector import tool_selector

load_dotenv()

import uuid
from typing import Any, List, Optional
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult

class SanitizedChatOpenAI(ChatOpenAI):
    def _generate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 부모 클래스의 _generate 호출
        chat_result = super()._generate(messages, stop, run_manager, **kwargs)
        
        # 결과 검사 및 수정
        if chat_result.generations:
            for generation in chat_result.generations:
                message = generation.message
                
                # 1. message.tool_calls 수정 (LangChain 내부용)
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if not tool_call.get('id'):
                            new_id = f"call_{str(uuid.uuid4()).replace('-', '')[:24]}"
                            tool_call['id'] = new_id
                            # print(f"⚠️ [SanitizedChatOpenAI] Fixed missing tool_call_id (Prop): {new_id}")

                # 2. message.additional_kwargs['tool_calls'] 수정 (OpenAI 전송용)
                if hasattr(message, 'additional_kwargs'):
                    raw_tool_calls = message.additional_kwargs.get('tool_calls', [])
                    if raw_tool_calls:
                        for raw_tc in raw_tool_calls:
                            if not raw_tc.get('id'):
                                new_id = f"call_{str(uuid.uuid4()).replace('-', '')[:24]}"
                                raw_tc['id'] = new_id
                                # print(f"⚠️ [SanitizedChatOpenAI] Fixed missing tool_call_id (Raw): {new_id}")
        
        return chat_result

    async def _agenerate(
        self,
        messages: List[Any],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 비동기 호출에 대해서도 동일하게 처리
        chat_result = await super()._agenerate(messages, stop, run_manager, **kwargs)
        
        if chat_result.generations:
            for generation in chat_result.generations:
                message = generation.message
                
                # 1. message.tool_calls 수정
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        if not tool_call.get('id'):
                            new_id = f"call_{str(uuid.uuid4()).replace('-', '')[:24]}"
                            tool_call['id'] = new_id
                            # print(f"⚠️ [SanitizedChatOpenAI] Fixed missing tool_call_id (Async Prop): {new_id}")

                # 2. message.additional_kwargs['tool_calls'] 수정
                if hasattr(message, 'additional_kwargs'):
                    raw_tool_calls = message.additional_kwargs.get('tool_calls', [])
                    if raw_tool_calls:
                        for raw_tc in raw_tool_calls:
                            if not raw_tc.get('id'):
                                new_id = f"call_{str(uuid.uuid4()).replace('-', '')[:24]}"
                                raw_tc['id'] = new_id
                                # print(f"⚠️ [SanitizedChatOpenAI] Fixed missing tool_call_id (Async Raw): {new_id}")
        
        return chat_result

class MatchMealCoach:
    def __init__(self):
        # 1. Fast LLM (Tool Selection, Chat) -> SanitizedChatOpenAI 적용
        self.fast_llm = SanitizedChatOpenAI(
            model="gpt-4.1-mini",
            temperature=1,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )

        # 2. Heavy LLM (Complex Reasoning) -> SanitizedChatOpenAI 적용
        self.heavy_llm = SanitizedChatOpenAI(
            model="gpt-5.2", 
            temperature=0.7, # 안정성을 위해 약간 낮춤
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            streaming=True,
            max_tokens=2048 # 충분한 출력 길이를 보장
        )
        
        # 전체 도구 리스트 (Map for Selection)
        self.all_tools = [
            analyze_health_and_nutrition, 
            recommend_food_from_db,
            calculate_exercise_burn,
            compare_foods,
            generate_shopping_list,
            recommend_seasonal_food,
            recommend_food_for_symptom,
            get_recipe_procedure,
            check_food_compatibility,
            calculate_maintenance_calories,
            suggest_healthy_alternative,
            calculate_water_needs,
            recommend_snack,
            analyze_nutrient_deficiency
        ]
        self.tools_map = {tool.name: tool for tool in self.all_tools}
        
        # 페르소나 정의
        self.PERSONA_PROMPTS = {
            "coach": "친절하고 전문적인 영양 조언을 제공하는 AI 전문가입니다. 사용자를 존중하며 공손한 말투(존댓말)를 사용하세요.",
            "friend": "30년 지기 '찐친'입니다. 격식 없이 편안한 반말을 사용하세요. 거친 농담과 유머를 섞어 대화하지만, 영양 정보만큼은 친구를 위해 진심으로 정확하게 조언해주세요. (예: '야, 그정도 먹었으면 이제 좀 굶어라', '이건 몸에 안 좋으니까 먹지 마라 좀')"
        }
        
        # 시스템 프롬프트 (Heavy/Fast 공용 구조, 상황에 따라 다를 수 있음)
        self.system_prompt_template = """
            당신은 '냠냠코치'입니다. 사용자의 [건강 프로필]과 [식사 기록]을 분석하여, {persona_instruction}

            [사용자 프로필]
            - 기본 정보: {age}세 / {gender} / {height}cm / {weight}kg
            - 신체 지수: BMI {bmi} ({bmi_status})
            - 보유 질환: {diseases}
            - 알레르기: {allergies}
            - 식성/취향: {flavors}

            ---
            [답변 형식 가이드 (필수 준수)]
            1. 모든 답변은 사용자가 핵심을 먼저 파악할 수 있도록 **3줄 요약** 섹션으로 시작하세요.
            2. **3줄 요약**이 끝난 후에는 반드시 `---` (대시 3개)를 입력하여 요약과 상세 내용을 구분해주세요.
            3. 상세 내용에서는 분석 결과와 함께 구체적인 개선 방향을 제안하세요.
            
            [형식 예시]
            **3줄 요약**
            1. (핵심 내용 1)
            2. (핵심 내용 2)
            3. (핵심 내용 3)

            ---

            (이후 상세 답변 작성...)
            
            ---
            [대화 컨텍스트]
            최근 대화 내용을 확인하고 흐름에 맞게 답변하세요:
            {history}
            
            ---
            [임무 가이드]
            1. **도구 활용:** 제공된 도구가 있다면 적극 활용하세요. 
               - 특히 '기간별 식단 분석' 요청 시에는 `analyze_health_and_nutrition` 도구를 사용하여 기초대사량과 권장 섭취량을 계산하고 비교 결과에 기반해 조언하세요.
               - 내부 함수명(예: analyze_health...)은 절대 답변에 노출하지 마세요.
            2. **데이터 분석:** 섭취한 메뉴 목록과 영양 통계를 꼼꼼히 살피세요. 부족하거나 과잉된 영양소(나트륨, 당류 등)를 지적하세요.
            3. **안전:** 사용자의 알레르기나 질병 정보를 최우선으로 고려하세요.
            4. **어조:** 선택된 페르소나에 맞춰 일관된 말투를 유지하세요.
            """
            
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt_template),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

    async def stream_agent_response(self, context_str: str, profile: dict, history: list = [], flavors: list = [], use_fast_model: bool = False, persona: str = "coach"):
        """
        제너레이터 함수: 답변을 스트리밍으로 yield 합니다.
        """
        # History 포맷팅
        history_text = ""
        for h in history:
            role = "사용자" if h.get("role") == "user" else "AI"
            history_text += f"- {role}: {h.get('content')}\n"

        # 0. Partial Prompt 준비
        persona_instruction = self.PERSONA_PROMPTS.get(persona, self.PERSONA_PROMPTS["coach"])
        
        partial_prompt = self.prompt.partial(
            persona_instruction=persona_instruction,
            age=profile.get('age', 0),
            gender=profile.get('gender', 'Unknown'),
            height=profile.get('height_cm', 170.0),
            weight=profile.get('weight_kg', 60.0),
            bmi=profile.get('bmi', 0.0),
            bmi_status=profile.get('bmi_status', 'Unknown'),
            diseases=profile.get('diseases') or "없음",
            allergies=profile.get('allergies') or "없음",
            flavors=", ".join(flavors) if flavors else "지정 안 함",
            history=history_text if history_text else "없음"
        )

        # 1. 도구 선별 (Vector Search + Fast LLM)
        # 모든 요청에 대해 도구 선별을 수행해 Context 최적화
        try:
            selected_tool_names = tool_selector.select_tools(context_str, self.tools_map)
        except Exception as e:
            print(f"Tool Selection Failed: {e}")
            selected_tool_names = []

        selected_tools = [self.tools_map[name] for name in selected_tool_names if name in self.tools_map]
        
        # 2. 모델 선택 및 실행 전략
        # - use_fast_model=True (Chat): Fast LLM 사용. 도구가 없으면 Chain으로, 있으면 Agent로.
        # - use_fast_model=False (Analysis): Heavy LLM 사용.
        
        # 도구가 없는데 Heavy Model을 써야 하는 경우? (심층 추론 필요 시) -> 분석 모드면 Heavy Model.
        # 도구가 있는데 Fast Model을 써야 하는 경우? (가벼운 검색 등) -> 가능.
        
        llm_to_use = self.fast_llm if use_fast_model else self.heavy_llm
        
        # 3. Agent Execution (Streaming)
        if not selected_tools:
            # 도구 없음 -> 단순 LLM Chain (Streaming)
            # AgentExecutor 없이 바로 stream
            print(f"🚀 Running {'FAST' if use_fast_model else 'HEAVY'} Chain (No Tools)")
            chain = partial_prompt | llm_to_use
            async for chunk in chain.astream({"input": context_str}):
                if chunk.content:
                    yield chunk.content
        else:
            # 도구 있음 -> AgentExecutor (Streaming)
            print(f"🛠️ Running {'FAST' if use_fast_model else 'HEAVY'} Agent with tools: {selected_tool_names}")
            agent = create_tool_calling_agent(llm_to_use, selected_tools, partial_prompt)
            executor = AgentExecutor(
                agent=agent, 
                tools=selected_tools, 
                verbose=True, 
                handle_parsing_errors=True,
                max_iterations=25 # 식단표 등 복잡한 작업 위해 반복 횟수 상향
            )
            
            try:
                # astream_events를 사용하여 'on_chat_model_stream' 이벤트만 필터링하여 yield
                async for event in executor.astream_events({"input": context_str}, version="v1"):
                    kind = event["event"]
                    
                    # LLM이 스트리밍하는 토큰 중 '최종 답변'에 해당하는 것만 추출해야 함.
                    if kind == "on_chat_model_stream":
                        # 데이터 구조 안전하게 접근
                        data = event.get("data", {})
                        chunk = data.get("chunk")
                        
                        # chunk가 있고, tool_calls가 없는(순수 텍스트) 경우만 yield
                        if chunk and hasattr(chunk, 'content') and chunk.content:
                            # tool_call_chunks가 있으면(즉, 도구 호출 중이면) 건너뜀
                            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                                continue
                                
                            yield chunk.content
            except Exception as e:
                # OpenAI 400 Error (Invalid type) 등을 포착하여 사용자에게 알림
                print(f"Agent Execution Error: {e}")
                yield f"\n\n[시스템 알림] 죄송합니다. 답변 생성 중 일시적인 오류가 발생했습니다. (Error: {str(e)[:50]}...)"
                # 로깅 또는 재시도 로직 추가 가능

coach = MatchMealCoach()