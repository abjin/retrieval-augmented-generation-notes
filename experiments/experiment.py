import os
import time
from google.genai import types
from google import genai
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

client = genai.Client()
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("rag-notes")


def embed_query(user_input):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=user_input,
        config=types.EmbedContentConfig(output_dimensionality=1024)
    )
    [embedding_obj] = result.embeddings
    return embedding_obj.values


def retrieve_relevant_documents(user_input, top_k=3, namespace=""):
    result = index.query(
        vector=embed_query(user_input),
        top_k=top_k,
        namespace=namespace,
        include_metadata=True
    )
    return result.matches


def run_rag_experiment(user_input="쿠버네티스가 무엇인가요?", top_k=3, namespace="cs256-ov0"):
    start_time = time.time()

    retrieved_docs = retrieve_relevant_documents(
        user_input, top_k=top_k, namespace=namespace)
    retrieved_texts = [doc.metadata['text'] for doc in retrieved_docs]

    context_parts = []
    for idx, text in enumerate(retrieved_texts, 1):
        context_parts.append(f"[문서 {idx}]\n{text}")

    prompt = f"""
        당신은 제공된 컨텍스트를 기반으로 정확하게 답변하는 AI 어시스턴트입니다.
        
        ## 규칙
        1. 반드시 아래 제공된 컨텍스트의 정보만을 사용하여 답변하세요.
        2. 컨텍스트에 없는 정보는 절대 추측하거나 외부 지식을 사용하지 마세요.
        3. 답변에 불확실한 부분이 있거나 컨텍스트에 정보가 부족한 경우, "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요.
        4. 가능한 경우 어떤 문서의 정보를 참조했는지 언급하세요 (예: "문서 1에 따르면...").

        ## 컨텍스트
        {"\n".join(context_parts)}

        ## 질문
        {user_input}
    """

    response = client.models.generate_content_stream(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    ttft = None
    last_chunk = None
    result = ""

    for chunk in response:
        if ttft is None:
            ttft = time.time() - start_time
        result += chunk.text
        last_chunk = chunk

    end_time = time.time()

    print({
        "prompt_tokens": last_chunk.usage_metadata.prompt_token_count,
        "completion_tokens": last_chunk.usage_metadata.candidates_token_count,
        "total_tokens": last_chunk.usage_metadata.total_token_count,
        "ttft": ttft,
        "latency": end_time - start_time,
        "response": result,
        "retrieved_texts": retrieved_texts
    })


if __name__ == "__main__":
    run_rag_experiment()
