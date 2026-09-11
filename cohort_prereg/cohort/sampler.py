"""Torch-free episode sampler — the SINGLE sampling implementation shared
by the trainer (scripts/cohort_society.py) and the admission fixture
(scripts/cohort_admission_fixture.py). Any drift between trainer and
fixture would falsely fail box admission, so neither may re-implement it.

Consumes ONLY the passed rng (random.Random(order_seed)); draw order is
part of the frozen protocol: r, x, then per-branch draws. Identical across
the five conditions at fixed order seed (the common semantic stream)."""

MIX = (0.10, 0.35, 0.675)
N_TRAIN_TPL = 24


def sample_spans(rng, split, train_phr, apply_chain, value_span,
                 forward_span, counts=None):
    def bump(key, val):
        if counts is not None:
            d = counts[key]
            d[str(val)] = d.get(str(val), 0) + 1

    r = rng.random()
    x = rng.randrange(17)
    if r < MIX[0]:
        spans = [value_span.format(x=x)] + [forward_span] * 3
        meta = ((), x, (), None)
        bump("depth", 0)
    elif r < MIX[1]:
        op = rng.randrange(12)
        tpl = rng.randrange(N_TRAIN_TPL)
        pos = rng.randrange(1, 4)
        spans = [value_span.format(x=x)] + [forward_span] * 3
        spans[pos] = train_phr[op][tpl]
        meta = ((op,), x, (tpl,), pos)
        bump("depth", 1); bump("op", op)
        bump("l1_pos", pos); bump("template", tpl)
    else:
        bank = split["train2"] if r < MIX[2] else split["train3"]
        seq = tuple(bank[rng.randrange(len(bank))])
        tpls = tuple(rng.randrange(N_TRAIN_TPL) for _ in seq)
        spans = [value_span.format(x=x)]
        spans += [train_phr[o][t] for o, t in zip(seq, tpls)]
        while len(spans) < 4:
            spans.append(forward_span)
        meta = (seq, x, tpls, None)
        bump("depth", len(seq))
        for o in seq: bump("op", o)
        for t in tpls: bump("template", t)
    bump("value", x)
    return spans, apply_chain(list(meta[0]), x), meta
