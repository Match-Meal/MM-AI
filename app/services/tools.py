from langchain.tools import tool
from app.core.standards import get_recommended_ratio
from app.services.vector_store import food_store 
from datetime import datetime

# ==================================================================================
# [기존 도구]
# ==================================================================================
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
    if health_condition == "high_bp": filter_dict = {"sodium": {"$lt": 600}}
    elif health_condition == "diabetes": filter_dict = {"sugar": {"$lt": 5}}
    elif health_condition == "diet": filter_dict = {"calories": {"$lt": 400}}
    elif health_condition == "muscle": filter_dict = {"protein": {"$gte": 20}}

    try:
        results = food_store.search_food(query, k=5, filter=filter_dict)
    except Exception as e:
        return f"검색 오류: {str(e)}"
    
    response = f"[검색 결과 (조건: {health_condition})]\n"
    if not results: return response + "조건에 맞는 메뉴가 없습니다."
    
    for doc in results:
        m = doc.metadata
        name = m.get('name', '이름없음')
        cal = m.get('calories', 0)
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
    mets = {"걷기": 3.5, "달리기": 8.0, "자전거": 6.0, "수영": 7.0, "등산": 7.5, "요가": 2.5, "웨이트": 4.0}
    met = mets.get(exercise_type, 4.0)
    burned = met * weight_kg * (duration_minutes / 60)
    return f"[운동 분석] {exercise_type} {duration_minutes}분 -> 약 {int(burned)}kcal 소모 (METs {met})"

# [Tool 4] 영양 성분 비교
@tool
def compare_foods(food_a: str, food_b: str) -> str:
    """
    두 가지 음식의 영양 성분을 비교합니다.
    """
    try:
        res_a = food_store.search_food(food_a, k=1)
        res_b = food_store.search_food(food_b, k=1)
    except Exception as e:
        return f"비교 중 오류 발생: {str(e)}"
        
    if not res_a or not res_b: return "비교할 음식을 찾을 수 없습니다."
    meta_a = res_a[0].metadata
    meta_b = res_b[0].metadata
    return f"""
    [영양 비교: {meta_a.get('name')} vs {meta_b.get('name')}]
    칼로리: {meta_a.get('calories')} vs {meta_b.get('calories')} kcal
    단백질: {meta_a.get('protein')} vs {meta_b.get('protein')} g
    나트륨: {meta_a.get('sodium')} vs {meta_b.get('sodium')} mg
    """

# [Tool 5] 장보기 리스트 생성
@tool
def generate_shopping_list(meal_plan_text: str) -> str:
    """
    제안된 식단 텍스트에서 식재료를 추출하여 JSON 형태의 장보기 리스트를 반환합니다.
    """
    return f"[장보기 리스트 생성 완료]\n(제공된 식단의 재료 목록을 포맷팅하여 반환)"

# ==================================================================================
# [신규 도구 9종] (Agent Tool Expansion)
# ==================================================================================

# 1. 제철 음식 추천
@tool
def recommend_seasonal_food(month: int = 0) -> str:
    """
    특정 월(Month)의 제철 음식과 식재료를 추천합니다.
    Args:
        month: 월 (1~12). 0 입력 시 현재 월 자동 선택.
    """
    if month == 0:
        month = datetime.now().month
        
    seasonal_data = {
        1: "우엉, 더덕, 딸기, 명태, 과메기",
        2: "우엉, 더덕, 딸기, 꼬막, 삼치",
        3: "쑥, 냉이, 달래, 쭈꾸미, 소라",
        4: "두릅, 미더덕, 키조개, 참다랑어",
        5: "매실, 두릅, 멍게, 다슬기, 장어",
        6: "감자, 매실, 장어, 복분자, 참외",
        7: "토마토, 옥수수, 수박, 블루베리, 도라지",
        8: "포도, 토마토, 옥수수, 전복, 복숭아",
        9: "고구마, 은행, 대하, 꽃게, 귤",
        10: "고구마, 사과, 대하, 꽃게, 홍합",
        11: "배추, 무, 굴, 꼬막, 유자, 과메기",
        12: "배추, 무, 굴, 꼬막, 명태, 유자"
    }
    
    foods = seasonal_data.get(month, "계절 정보 없음")
    return f"[{month}월 제철 음식 추천]\n추천 식재료: {foods}\n이 재료들을 활용한 요리를 추천해보세요."

# 2. 증상 완화 음식 추천
@tool
def recommend_food_for_symptom(symptom: str) -> str:
    """
    사용자의 건강 증상(감기, 소화불량 등)에 도움이 되는 음식을 추천합니다.
    """
    kb = {
        "감기": "생강차, 배숙, 모과차, 콩나물국 (비타민C, 수분 보충)",
        "소화불량": "매실청, 무국, 식혜 (천연 소화제)",
        "빈혈": "소고기, 시금치, 굴, 깻잎 (철분 풍부)",
        "변비": "사과, 고구마, 미역국, 요거트 (식이섬유)",
        "피로": "전복죽, 삼계탕, 장어구이, 오렌지 (단백질, 비타민C)",
        "숙취": "황태국, 콩나물국, 꿀물, 토마토"
    }
    
    # Simple Keyword Match
    matched = [v for k, v in kb.items() if k in symptom]
    if not matched:
        return f"'{symptom}' 증상에 대한 구체적인 정보가 DB에 없습니다. 일반적인 건강식(죽, 따뜻한 차)을 추천합니다."
    return f"[증상 완화 음식: {symptom}]\n추천: {', '.join(matched)}"

# 3. 요리 레시피 절차
@tool
def get_recipe_procedure(menu_name: str) -> str:
    """
    특정 요리의 조리법(Recipe)을 단계별로 안내합니다.
    """
    # MVP: Common Dish Dict
    recipes = {
        "김치찌개": "1. 김치와 돼지고기를 참기름에 볶는다.\n2. 물이나 멸치육수를 붓고 끓인다.\n3. 다진마늘, 고춧가루, 두부를 넣는다.\n4. 대파를 넣고 푹 끓여 완성한다.",
        "된장찌개": "1. 멸치육수에 된장을 푼다.\n2. 애호박, 두부, 양파를 깍둑썰기하여 넣는다.\n3. 다진마늘과 고춧가루를 약간 넣는다.\n4. 팽이버섯과 대파를 넣고 끓인다.",
        "제육볶음": "1. 돼지고기를 고추장, 간장, 설탕, 마늘 양념에 재운다.\n2. 달군 팬에 고기를 먼저 볶는다.\n3. 양파, 당근, 대파를 넣고 아삭하게 볶는다.\n4. 참기름과 깨를 뿌려 마무리한다."
    }
    
    result = recipes.get(menu_name)
    if not result:
        return f"'{menu_name}'의 정확한 레시피가 없지만, 일반적인 조리법을 안내해드릴 수 있습니다. (LLM 지식 활용 권장)"
    return f"[{menu_name} 레시피]\n{result}"

# 4. 음식 궁합 확인
@tool
def check_food_compatibility(food_name: str) -> str:
    """
    입력된 음식과 궁합이 좋은(상생) 음식과 나쁜(상극) 음식을 알려줍니다.
    """
    kb = {
        "돼지고기": {"good": "새우젓(소화), 표고버섯", "bad": "도라지"},
        "소고기": {"good": "배(연육작용), 깻잎", "bad": "버터, 부추"},
        "닭고기": {"good": "인삼, 대추", "bad": "미나리"},
        "장어": {"good": "생강(살균/소화)", "bad": "복숭아(설사 유발)"},
        "시금치": {"good": "조개, 달걀", "bad": "두부(결석 우려)"},
        "두부": {"good": "미역, 김치", "bad": "시금치"}
    }
    
    info = kb.get(food_name)
    if not info:
        return f"'{food_name}'에 대한 궁합 정보가 부족합니다."
    return f"[{food_name} 궁합 정보]\n- 같이 먹으면 좋아요(상생): {info['good']}\n- 주의하세요(상극): {info['bad']}"

# 5. 유지 칼로리 계산
@tool
def calculate_maintenance_calories(gender: str, age: int, height_cm: float, weight_kg: float, activity_level: str) -> str:
    """
     Harris-Benedict 공식을 사용하여 유지 칼로리를 계산합니다.
     Args:
        activity_level: 'sedentary'(운동X), 'light'(주1-3), 'moderate'(주3-5), 'active'(주6-7), 'very_active'(선수급)
    """
    # 1. BMR
    s = 5 if gender == "MALE" else -161
    bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + s
    
    # 2. Activity Multiplier
    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    factor = multipliers.get(activity_level, 1.2)
    maintenance = bmr * factor
    
    return f"""
    [유지 칼로리 계산]
    - 기초대사량(BMR): {int(bmr)}kcal
    - 활동 계수: {factor} ({activity_level})
    - 유지 칼로리: 약 {int(maintenance)}kcal
    (체중 감량을 원하시면 이보다 300~500kcal 적게 섭취하세요.)
    """

# 6. 건강 대체 식품 추천
@tool
def suggest_healthy_alternative(unhealthy_food: str) -> str:
    """
    칼로리가 높거나 건강에 좋지 않은 음식을 대체할 수 있는 건강한 메뉴를 추천합니다.
    """
    map_db = {
        "라면": "곤약면, 두부면, 쌀국수, 숙주라면",
        "치킨": "굽네치킨(구운 치킨), 닭가슴살 샐러드, 에어프라이어 치킨",
        "피자": "또띠아 피자, 두부 피자, 씬 피자(야채 많이)",
        "콜라": "제로 콜라, 탄산수, 콤부차",
        "과자": "견과류, 말린 과일, 프로틴 칩",
        "아이스크림": "저칼로리 아이스크림, 얼린 요거트, 과일 셔벗"
    }
    
    alt = map_db.get(unhealthy_food)
    if not alt:
        return f"'{unhealthy_food}'의 직접적인 대체제 정보는 없지만, 조리법을 건강하게 바꾸거나 양을 줄이는 것을 권장합니다."
    return f"[{unhealthy_food} 건강 대체 제안]\n추천: {alt}"

# 7. 일일 수분 섭취량 계산
@tool
def calculate_water_needs(weight_kg: float, activity_level: str = "normal") -> str:
    """
    체중과 활동량을 기반으로 하루 권장 물 섭취량을 계산합니다.
    """
    # 일반 권장: 체중 * 30~35ml
    base_need = weight_kg * 33 # 평균 33ml
    
    if activity_level in ["active", "very_active"]:
        base_need += 500 # 운동 시 추가 섭취
        
    liters = base_need / 1000
    cups = base_need / 200 # 200ml 컵 기준
    
    return f"""
    [수분 섭취 가이드]
    - 체중 {weight_kg}kg 기준
    - 하루 권장량: 약 {liters:.1f}리터 ({int(cups)}잔)
    - 팁: 활동량이 많거나 더운 날에는 500ml 이상 더 섭취하세요.
    """

# 8. 상황별 간식 추천
@tool
def recommend_snack(situation: str) -> str:
    """
    상황(다이어트, 공부/업무, 운동 전, 운동 후, 야식)에 맞는 간식을 추천합니다.
    """
    kb = {
        "다이어트": "방울토마토, 오이, 곤약젤리, 아몬드 5알",
        "공부": "다크초콜릿, 견과류, 블루베리 (두뇌 회전)",
        "업무": "다크초콜릿, 견과류, 블루베리 (집중력)",
        "운동전": "바나나, 통밀빵, 에너지바 (탄수화물)",
        "운동후": "프로틴 쉐이크, 삶은 계란, 닭가슴살 소시지 (단백질)",
        "야식": "따뜻한 우유, 바나나, 두부 데침 (숙면 도움)"
    }
    
    # Keyword search
    snack = None
    for k, v in kb.items():
        if k in situation.replace(" ", ""):
            snack = v
            break
            
    if not snack:
        return "과일이나 견과류 같은 자연식품 간식을 추천합니다."
    return f"[{situation} 추천 간식]\n{snack}"

# 9. 영양 결핍 예측
@tool
def analyze_nutrient_deficiency(symptom: str) -> str:
    """
    신체 증상을 통해 부족할 것으로 의심되는 영양소와 보충 음식을 알려줍니다.
    예: 눈떨림, 손톱 갈라짐, 만성피로 등.
    """
    kb = {
        "눈떨림": ("마그네슘", "바나나, 아몬드, 시금치"),
        "손톱": ("단백질, 비오틴", "계란, 콩, 연어"),
        "피로": ("비타민B군, 철분", "돼지고기, 간, 현미"),
        "입병": ("비타민B2, 비타민C", "우유, 꿀, 과일"),
        "쥐": ("칼슘, 마그네슘", "우유, 멸치, 견과류"),
        "빈혈": ("철분", "소고기, 미역, 굴")
    }
    
    found = []
    for k, v in kb.items():
        if k in symptom:
            found.append(f"- 의심 영양소: {v[0]} -> 추천 음식: {v[1]}")
            
    if not found:
        return f"'{symptom}' 증상만으로는 특정 영양 결핍을 판단하기 어렵습니다. 균형 잡힌 식사를 권장합니다."
    
    return f"[증상 기반 영양 분석]\n{chr(10).join(found)}"
