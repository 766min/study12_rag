<aside>
<img src="notion://custom_emoji/83c75a39-3aba-4ba4-a792-7aefe4b07895/30c2dc3e-f514-8052-98d5-007a88daa225" alt="notion://custom_emoji/83c75a39-3aba-4ba4-a792-7aefe4b07895/30c2dc3e-f514-8052-98d5-007a88daa225" width="40px" />

**목차**

</aside>

---

## 1. **LCEL과 Runnable 프로토콜의 등장 배경**

### 1.1 **레거시 체인(LLMChain)의 한계**

LangChain 초기 버전의 체인 구현체들은 다음과 같은 구조적 결함을 안고 있었습니다.

- **비일관된 입출력 인터페이스**: 체인마다 실행 메서드가 제각각(`run()`, `predict()`, `__call__()` 등).
- **스트리밍/비동기 지원의 분편화**: 실시간 UI를 위한 토큰 스트리밍이나 고성능 비동기 IO 처리를 추가하려면 컴포넌트 전체를 재작성해야 함.
- **관측성(Observability) 부재**: 파이프라인 내부 중간 단계의 지연 시간, 입출력 값, 토큰 소비량을 단계별로 추적하기 어려움.

### **1.2 Runnable 프로토콜: 리눅스 파이프 철학의 계승**

LCEL은 유닉스/리눅스의 파이프 철학(<u>**"작고 명확한 프로그램을 만들고, 텍스트 스트림을 통해 서로 연결하라"**</u>)을 현대 LLM 오케스트레이션에 적용했습니다.

- **모든 것은 Runnable이다**: Prompt, Model, Tool, OutputParser, Retriever는 물론, 임의의 파이썬 함수와 딕셔너리까지 동일한 인터페이스 프로토콜을 따릅니다.
- **단일 인터페이스, 다중 실행 모드**: 동일한 파이프라인 코드로 단일 호출, 비동기 호출, 병렬 배치, 실시간 스트리밍을 별도 코드 수정 없이 일관되게 지원합니다.

## 2. **Runnable 표준 6대 메서드 인터페이스**

모든 Runnable 구현체는 입력(Input)과 출력(Output) 타입을 기반으로 표준 6대 메서드를 기본 제공합니다.

| 작업 유형 | 동기(Sync) 메서드 | 비동기(Async) 메서드 | 반환 타입 / 동작 특징 |
| --- | --- | --- | --- |
| 단일 실행 | .invoke(input, config=None) | .ainvoke(input, config=None) | Output 단일 결과 반환 |
| 배치 실행 | .batch(inputs, config=None) | .abatch(inputs, config=None) | List[Output], 내부 스레드풀/비동기 루프 병렬 처리 |
| 스트리밍 | .stream(input, config=None) | .astream(input, config=None) | Iterator[Output] / AsyncIterator[Output] |

<aside>
💡 **고급 스트리밍 이벤트 구독 (astream_events)**

---

체인 전체 결과뿐 아니라, 파이프라인 내부 노드(예: 모델의 생성 토큰, 리트리버의 문서 검색 완료 등)의 세부 이벤트를 실시간으로 수신할 때 사용합니다.

```python
# v2 이벤트 API 사용 예시
async for event in chain.astream_events(query, version="v2"):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        print(content, end="", flush=True)
```

</aside>

## 3. **파이프 연산자(|)의 내부 동작 원리**

LCEL의 상징인 `|` 문법은 파이썬의 연산자 오버로딩(Operator Overloading)을 통해 구현되어 있습니다.

!image.png

### 3.1 **__or__ 와 __ror__**

- **Runnable.__or__(self, other)**: 좌항이 Runnable일 때 우항 other를 체인으로 연결합니다.
- **Runnable.__ror__(self, other)**: 좌항 other가 Runnable이 아닐 때(예: 순수 파이썬 딕셔너리), 우항인 self가 좌항을 가로채 연결합니다.

### 3.2 **암묵적 강제 변환 (coerce_to_runnable)**

LangChain은 | 연산자 평가 시 피연산자의 타입을 검사하여 자동으로 내부 Runnable 래퍼로 변환합니다.

```python
# 원본 코드
chain = {"question": RunnablePassthrough()} | prompt | model

# 내부 변환 과정
# 1. {"question": ...}은 dict -> RunnableParallel로 변환
# 2. prompt.__ror__(dict) 가 트리거됨
# 3. 최종적으로 RunnableSequence(RunnableParallel(...), prompt, model) 생성
```

- **dict** $\rightarrow$ RunnableParallel
- **Callable (함수/람다)** $\rightarrow$ RunnableLambda
- **Runnable**  $\rightarrow$ 그대로 유지
****

## 4. **핵심 프리미티브(Primitives) 상세 분석**

### **4.1 RunnablePassthrough & RunnablePassthrough.assign**

데이터 파이프라인에서 입력을 유지하거나 필드를 증강할 때 사용합니다.

```python
from langchain_core.runnables import RunnablePassthrough

# 원본 그대로 전달
# 입력: "양자역학이란?" -> 출력: {"context": "...", "question": "양자역학이란?"}
setup_and_retrieval = {
    "context": retriever,
    "question": RunnablePassthrough()
}

# assign을 통한 필드 증강 (기존 dict 보존)
# 입력: {"question": "..."}
# 출력: {"question": "...", "context": [...]}
chain_with_assign = RunnablePassthrough.assign(
    context=lambda x: retriever.invoke(x["question"])
)
```

### **4.2 RunnableParallel**

여러 독립적인 작업을 병렬로 수행하고 결과를 단일 딕셔너리로 결합합니다.

동기 `.batch()`나 비동기 환경에서 각 키의 작업들이 내부적으로 멀티스레딩/태스크로 동시 처리되어 전체 지연 시간을 단축시킵니다.

### 4.3 **@chain 데코레이터와 RunnableLambda**

복잡한 절차적 로직(조건문, 루프, 예외 처리)을 파이썬 네이티브 함수로 작성하면서도 Runnable 규격을 유지하는 최적의 방법입니다.

```python
from langchain_core.runnables import chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

model = init_chat_model(
		model="gpt-6-luna",
		model_provider="openai",
		reasoning_effort="low"
)

@chain
def routing_chain(input_data: dict):
    """조건문과 커스텀 비즈니스 로직을 순수 파이썬으로 구현"""
    topic = input_data.get("topic", "general")
    
    if topic == "technical":
        prompt = ChatPromptTemplate.from_template("기술 전문가 관점에서 답변하세요: {question}")
    else:
        prompt = ChatPromptTemplate.from_template("친절한 상담원 관점에서 답변하세요: {question}")
        
    sub_chain = prompt | model
    return sub_chain.invoke(input_data)

# @chain으로 생성된 객체는 invoke, astream, batch를 모두 지원
result = routing_chain.invoke({"topic": "technical", "question": "GC 동작 원리"})
```

## 5. **종합 실습: RAG 파이프라인 구축**

다음은 앞서 배운 모든 개념(연산자 오버로딩, `RunnablePassthrough.assign`, `@chain`, 스트리밍, Fallback)을 통합한 전체 코드 예제입니다.

```python
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, chain
from langchain.chat_models import init_chat_model

# 프롬프트 & 파서 정의
prompt = ChatPromptTemplate.from_messages([
    ("system", "아래 문맥(Context)을 기반으로 질문에 답하세요.\n\n[문맥]:\n{context}"),
    ("human", "{question}")
])
parser = StrOutputParser()

# 모델 선언 및 Fallback 구성
primary_model = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=0)
fallback_model = init_chat_model(model="gpt-6-luna", model_provider="openai", reasoning_effort="low")
robust_model = primary_model.with_fallbacks([fallback_model])

# 모의 검색기(Retriever)를 @chain으로 구현
@chain
def custom_retriever(query: str) -> str:
    # 실제 환경: vector_store.similarity_search(query)
    docs = [
        "LangChain의 LCEL은 파이프(|) 연산자를 통해 체인을 구성하는 선언적 문법입니다.",
        "Runnable은 invoke, stream, batch를 공통 인터페이스로 표준화합니다."
    ]
    return "\n".join(docs)

# 전체 파이프라인 조합 (LCEL)
rag_pipeline = (
    RunnablePassthrough.assign(
        context=lambda x: custom_retriever.invoke(x["question"])
    )
    | prompt
    | robust_model
    | parser
)

# 실행 검증 (동기 스트리밍)
if __name__ == "__main__":
    query_input = {"question": "LCEL과 Runnable의 핵심 관계가 뭐야?"}
    
    print("=== 실시간 스트리밍 답변 출력 ===")
    for chunk in rag_pipeline.stream(query_input):
        print(chunk, end="", flush=True)
    print("\n")
```

## **6. 핵심 요약 및 모범 사례**

<u>**절차적 복잡도는 @chain으로 해결하라.**</u>

삼항 연산자나 중첩 RunnableBranch로 LCEL을 무리하게 복잡하게 구성하기보다는, @chain 데코레이터를 붙인 순수 파이썬 함수를 작성하는 것이 가독성, 디버깅, 유지보수 측면에서 우수합니다.

<u>**사이드 이펙트는 딕셔너리로 감싸라.**</u>

이전 단계의 입력값이 체인 후반부까지 살아있어야 한다면 반드시 RunnablePassthrough.assign()을 활용하여 필드를 점진적으로 증강해 나가는 패턴을 취해야 합니다.

<u>**복원력(Resilience) 확보**</u>

상용 서비스 배포 시 API Rate Limit이나 네트워크 장애에 대비해 .with_fallbacks([대체_runnable]) 및 .with_retry(stop_after_attempt=3)를 필수로 고려합니다.

---

Copyright ⓒ TeamSparta All rights reserved.