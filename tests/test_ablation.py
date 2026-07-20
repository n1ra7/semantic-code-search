from eval.ablation import format_table


def test_format_table_renders_markdown():
    rows = [
        ("baseline (line, dense)", {"precision@k": 0.1, "recall@k": 0.5, "hit@k": 0.6, "mrr": 0.4, "ndcg@k": 0.39}),
        ("+ AST chunking", {"precision@k": 0.2, "recall@k": 0.6, "hit@k": 0.7, "mrr": 0.5, "ndcg@k": 0.48}),
    ]
    table = format_table(rows)
    assert "| Config |" in table
    assert "baseline (line, dense)" in table
    assert "+ AST chunking" in table
    # values are formatted to 3 decimals
    assert "0.100" in table and "0.480" in table
    # one header + one separator + one row per config
    assert len(table.splitlines()) == 2 + len(rows)
