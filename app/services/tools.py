from langchain.tools import tool
from app.core.standards import get_recommended_ratio
from app.services.vector_store import food_store 

# [Tool 1] 건강 상태 및 영양 분석
@tool
def analyze_health_and_nutrition(age: int = 30, gender: str = "MALE", height_cm: float = 170.0, weight_kg: float = 70.0, current_calories: float = 0.0, diseases: str = "없음", allergies: str = "없음") -> str:
    """
    사용자의 신체 정보(BMI, BMR)와 오늘 섭취량을 분석합니다.
    Args:
        age: 나이
        gender: 성별 (MALE/FEMALE)
        height_cm: 키 (cm)
        weight_kg: 몸무게 (kg)
        current_calories: 현재 섭취 칼로리
        diseases: 보유 질환
        allergies: 알레르기 정보
    """
    # BMR 계산 (Mifflin-St Jeor)
    s_val = 5 if gender == "MALE" else -161
    bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + s_val
    
    target_calories = bmr * 1.375 
    
    return f"""
    [신체 분석] BMR {int(bmr)}kcal, 일일 권장 {int(target_calories)}kcal
    [현재 상태] {int(current_calories)}kcal 섭취 (권장량의 {int(current_calories/target_calories*100) if target_calories else 0}%)
    [건강 정보] 질병: {diseases}, 알레르기: {allergies}
    """

# [Tool 2] 조건부 음식 추천 (RAG + Filter)
@tool
def recommend_food_from_db(query: str, health_condition: str = "general") -> str:
    """
    음식을 검색합니다. health_condition에 따라 필터링합니다.
    옵션: 'general', 'high_bp'(고혈압), 'diabetes'(당뇨), 'diet'(다이어트), 'muscle'(근성장)
    """
    filter_dict = {}
    
    if health_condition == "high_bp":     
        filter_dict = {"sodium": {"$lt": 600}}
    elif health_condition == "diabetes":  
        filter_dict = {"sugar": {"$lt": 5}}
    elif health_condition == "diet":      
        filter_dict = {"calories": {"$lt": 400}}
    elif health_condition == "muscle":    
        filter_dict = {"protein": {"$gte": 20}}

    try:
        results = food_store.search_food(query, k=5, filter=filter_dict)
    except Exception as e:
        return f"검색 오류: {str(e)}"
    
    response = f"[검색 결과 (조건: {health_condition})]\n"
    if not results:
        return response + "조건에 맞는 메뉴가 없습니다."
    
    for doc in results:
        m = doc.metadata
        name = m.get('name', '이름없음')
        cal = m.get('calories', 0)
        # 나트륨/당류 정보는 조건에 따라 강조
        detail = ""
        if health_condition == "high_bp": detail = f", 나트륨 {m.get('sodium',0)}mg"
        if health_condition == "diabetes": detail = f", 당류 {m.get('sugar',0)}g"
        
        response += f"- {name} ({cal}kcal{detail})\n"
    
    return response

# [Tool 3] 운동 칼로리 계산 (METs)
@tool
def calculate_exercise_burn(weight_kg: float, exercise_type: str, duration_minutes: int) -> str:
    """
    사용자의 체중과 운동 종류, 시간을 입력받아 소모 칼로리를 계산합니다.
    Args:
        weight_kg: 체중(kg)
        exercise_type: 운동 종류 (걷기, 달리기, 자전거, 수영, 등산, 요가, 웨이트)
        duration_minutes: 운동 시간(분)
    """
    # METs Table (Approximate)
    mets = {
        "걷기": 3.5,
        "달리기": 8.0,
        "자전거": 6.0,
        "수영": 7.0,
        "등산": 7.5,
        "요가": 2.5,
        "웨이트": 4.0
    }
    
    met = mets.get(exercise_type, 4.0) # 기본값: 보통 강도 운동
    
    # 칼로리 소모량 = MET * 3.5 * weight(kg) / 200 * duration(min)
    # or more simply: MET * weight * duration_hours
    burned = met * weight_kg * (duration_minutes / 60)
    
    return f"""
    [운동 분석]
    - 운동 종류: {exercise_type}
    - 시간: {duration_minutes}분
    - 예상 소모 칼로리: 약 {int(burned)}kcal
    (METs {met} 기준 추정치입니다.)
    """

# [Tool 4] 영양 성분 비교
@tool
def compare_foods(food_a: str, food_b: str) -> str:
    """
    두 가지 음식의 영양 성분을 비교합니다.
    Args:
        food_a: 비교할 첫 번째 음식
        food_b: 비교할 두 번째 음식
    """
    try:
        res_a = food_store.search_food(food_a, k=1)
        res_b = food_store.search_food(food_b, k=1)
    except Exception as e:
        return f"비교 중 오류 발생: {str(e)}"
        
    if not res_a or not res_b:
        return "비교할 음식을 찾을 수 없습니다."
        
    meta_a = res_a[0].metadata
    meta_b = res_b[0].metadata
    
    # Simple comparison string
    return f"""
    [영양 비교: {meta_a.get('name')} vs {meta_b.get('name')}]
    1. 칼로리: {meta_a.get('calories')} vs {meta_b.get('calories')} kcal
    2. 탄수화물: {meta_a.get('carbohydrate')} vs {meta_b.get('carbohydrate')} g
    3. 단백질: {meta_a.get('protein')} vs {meta_b.get('protein')} g
    4. 지방: {meta_a.get('fat')} vs {meta_b.get('fat')} g
    5. 나트륨: {meta_a.get('sodium')} vs {meta_b.get('sodium')} mg
    """

import json

# [Tool 5] 장보기 리스트 생성
@tool
def generate_shopping_list(meal_plan_text: str) -> str:
    """
    제안된 식단 텍스트에서 식재료를 추출하여 JSON 형태의 장보기 리스트를 반환합니다.
    Args:
        meal_plan_text: 식단 텍스트 (예: "아침은 사과와 계란...")
    """
    # This acts as a formatting helper. The LLM calls this with its own output.
    # In a real agent, the LLM creates the inputs.
    # We can just return the text wrapped in a specific format for the frontend to parse,
    # or instructing the LLM to format it is often enough. 
    # But as a tool, we can simulate extracting key terms if we had an NLP model.
    # Here, we will just prompt the Agent to be precise via this tool's docstring.
    
    return f"""
    [장보기 리스트 생성 완료]
    제공된 식단을 바탕으로 필요한 재료를 정리했습니다.
    (실제로는 현재 텍스트 처리 로직이 없어 LLM이 추출한 내용을 포맷팅만 합니다.)
    """
