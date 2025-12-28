# rag-optimization-experiment

RAG 파라미터(TopK, Chunk Size, Overlap) 최적화 실험

## Contents

- `experiment.md` - 실험 결과 및 분석 문서
- `preprocessing_pipeline.py` - 문서 청킹 및 임베딩 파이프라인
- `experiment.py` - RAG 실험 실행 스크립트
- `evaluate.py` - RAGAS 메트릭 기반 성능 평가
- `questions.json` - 평가용 질문 셋
- `evaluation_results.json` - 평가 결과 데이터

## Setup

```bash
cd experiments
pip install -r requirements.txt
```

## Download Dataset

```bash 
git clone https://github.com/kubernetes/website
cp -r  ./website/content/ko ./ko
```