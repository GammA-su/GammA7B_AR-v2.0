from gamma7b_data.sampling import TokenAwareSampler


def test_sampler_deterministic_and_weighted():
    domain_weights = {"a": 0.7, "b": 0.3}
    source_weights = {"a": {"a1": 1.0}, "b": {"b1": 1.0}}

    sampler1 = TokenAwareSampler(domain_weights, source_weights, seed=42)
    sampler2 = TokenAwareSampler(domain_weights, source_weights, seed=42)

    seq1 = []
    seq2 = []
    for _ in range(50):
        d1 = sampler1.choose_domain()
        s1 = sampler1.choose_source(d1)
        sampler1.update(d1, s1, 10)
        seq1.append((d1, s1))

        d2 = sampler2.choose_domain()
        s2 = sampler2.choose_source(d2)
        sampler2.update(d2, s2, 10)
        seq2.append((d2, s2))

    assert seq1 == seq2
    count_a = sum(1 for d, _ in seq1 if d == "a")
    count_b = sum(1 for d, _ in seq1 if d == "b")
    assert count_a > count_b
    assert abs((count_a / (count_a + count_b)) - 0.7) < 0.2
