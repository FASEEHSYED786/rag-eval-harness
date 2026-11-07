from rag_eval import recall_at_k, mrr, ndcg_at_k, faithfulness, evaluate


def test_basic_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 3) == 0.5
    assert mrr(["a", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["a"], {"a"}, 3) == 1.0
    assert ndcg_at_k(["x"], {"a"}, 3) == 0.0


def test_faithfulness():
    assert faithfulness("leave is 25 days", "annual leave is 25 days") == 1.0
    assert faithfulness("pizza on mars", "annual leave") == 0.0


def test_evaluate():
    cases = [{"question": "q1", "relevant": ["d1"]}, {"question": "q2", "relevant": ["d2"]}]
    r = evaluate(cases, lambda q, k: ["d1"] if q == "q1" else ["x", "d2"])
    assert r["recall@k"] == 1.0 and r["mrr"] == 0.75
