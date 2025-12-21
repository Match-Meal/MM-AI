@router.post("/feedback")
async def get_feedback(request: DietFeedbackRequest):
    # 1. Spring Boot에서 받은 DTO를 텍스트로 요약
    current_intake_summary = f"""
    사용자 나이: 25세 (예시 데이터)
    오늘 총 섭취량:
    - 칼로리: {request.daily_total.total_calories}
    - 탄수화물: (계산필요 혹은 DTO에 추가필요)
    - 단백질: (계산필요 혹은 DTO에 추가필요)
    - 지방: (계산필요 혹은 DTO에 추가필요)
    """
    
    # 2. Agent 실행
    # Agent가 스스로 판단하여 Tool을 호출함
    result = nutrition_agent.run(current_intake_summary + "\n" + request.user_message)
    
    return {"answer": result["output"]}