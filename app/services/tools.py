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