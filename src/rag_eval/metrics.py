import math
import re

W = re.compile(r"[a-z0-9]+")


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def faithfulness(answer: str, context: str) -> float:
    """Share of answer tokens supported by the retrieved context (cheap hallucination signal)."""
    a, c = set(W.findall(answer.lower())), set(W.findall(context.lower()))
    return len(a & c) / len(a) if a else 0.0


def evaluate(cases: list[dict], retriever, k: int = 3) -> dict:
    """cases: [{'question', 'relevant': [ids]}]; retriever(question, k) -> list of doc ids."""
    n = len(cases) or 1
    r = m = g = 0.0
    for c in cases:
        got, rel = retriever(c["question"], k), set(c["relevant"])
        r += recall_at_k(got, rel, k)
        m += mrr(got, rel)
        g += ndcg_at_k(got, rel, k)
    return {"recall@k": r / n, "mrr": m / n, "ndcg@k": g / n, "cases": len(cases)}
