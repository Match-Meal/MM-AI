# 연령별 탄:단:지 권장 비율
NUTRITION_STANDARDS = {
    "19-29": {"carb": 0.55, "protein": 0.20, "fat": 0.25},
    "30-49": {"carb": 0.55, "protein": 0.20, "fat": 0.25},
    "50-64": {"carb": 0.60, "protein": 0.20, "fat": 0.20},
    "65+":   {"carb": 0.60, "protein": 0.20, "fat": 0.20},
}

def get_recommended_ratio(age: int):
    if age < 19: return NUTRITION_STANDARDS["19-29"] # 미성년자는 20대 기준 적용 (임시)
    if 19 <= age <= 29: return NUTRITION_STANDARDS["19-29"]
    if 30 <= age <= 49: return NUTRITION_STANDARDS["30-49"]
    if 50 <= age <= 64: return NUTRITION_STANDARDS["50-64"]
    return NUTRITION_STANDARDS["65+"]