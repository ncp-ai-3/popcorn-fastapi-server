from llama_cpp import Llama

# 1. 나영님의 로컬 LLM 불러오기
print("⏳ M4 GPU로 LLM을 깨우는 중...")
llm = Llama(
    model_path="./qwen2-1_5b-instruct-q4_k_m.gguf", 
    n_gpu_layers=-1, 
    verbose=False
)

# 2. 사용자 질문 (RAG 데이터 없음!)
query = "이번 주말 성수동에서 갈 만한 캐릭터 팝업스토어 추천해 줘."

# 3. 팝업 데이터 없이 순수하게 질문만 던지는 프롬프트
prompt = f"""<|im_start|>system
당신은 서울 성수동 팝업스토어 안내 가이드입니다.<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""

# 4. 답변 생성 및 출력
print("\n🤖 AI가 (인터넷 검색 없이) 기억만으로 답변을 적고 있습니다...\n")
output = llm(prompt, max_tokens=512, stop=["<|im_end|>"], echo=False)

# 결과 확인
print("--- ✅ RAG 없는 순수 AI의 답변 ---")
print(output["choices"][0]["text"].strip())