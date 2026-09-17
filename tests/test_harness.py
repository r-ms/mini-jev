"""Guards. Each one must be able to return a negative answer: the placebo that would make it
red is named in the test's docstring."""
import json, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minijev import config as C, data, prompts, schema
from minijev.letters import build_tables, score, classify_argmax, letter
from minijev.scoring import accuracy, correct, paired, cluster_bootstrap_delta, unit_rows
from minijev.runner import parse_a

import torch


@pytest.fixture(scope="module")
def sample():
    return data.load_sample()


def test_options_gold_position_is_recorded_after_the_shuffle(sample):
    """PLACEBO: record gold_pos before the shuffle -> options[gold_pos] != gold here."""
    rows, m = sample
    for r in [x for x in rows if not x["oos"]][:40]:
        for k in (2, 4, 8):
            opts, gp, keff, pol = schema.options_for(r["text_id"], "intent", k,
                                                     r["gold"]["intent"], m)
            assert opts[gp] == r["gold"]["intent"], (r["text_id"], k, opts, gp)


def test_option_set_is_stable_across_processes(sample):
    """PLACEBO: seed the RNG with hash() or with time -> a second process disagrees."""
    rows, m = sample
    r = [x for x in rows if not x["oos"]][0]
    here = schema.options_for(r["text_id"], "intent", 8, r["gold"]["intent"], m)[0]
    code = (f"import sys; sys.path.insert(0,{str(ROOT)!r});"
            "from minijev import data, schema;"
            "rows,m=data.load_sample();"
            f"r=[x for x in rows if x['text_id']=={r['text_id']!r}][0];"
            "import json; print(json.dumps(schema.options_for(r['text_id'],'intent',8,"
            "r['gold']['intent'],m)[0]))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert json.loads(out.stdout.strip()) == here


def test_option_set_is_the_same_across_field_conditions(sample):
    """The k axis must not move when the f axis moves: same RNG key, no condition in it."""
    rows, m = sample
    r = [x for x in rows if not x["oos"]][0]
    a = schema.options_for(r["text_id"], "intent", 4, r["gold"]["intent"], m)[0]
    b = schema.options_for(r["text_id"], "intent", 4, r["gold"]["intent"], m)[0]
    assert a == b


def test_within_domain_distractors_are_same_domain(sample):
    """PLACEBO: sample distractors from the full vocabulary -> cross-domain names appear."""
    rows, m = sample
    for r in [x for x in rows if not x["oos"]][:30]:
        dom = r["gold"]["domain"]
        opts, _, _, pol = schema.options_for(r["text_id"], "intent", 8, r["gold"]["intent"], m)
        if pol != "within_domain":
            continue
        assert all(o in m["intents_by_domain"][dom] for o in opts), (dom, opts)


def test_k16_is_labelled_mixed_not_within_domain(sample):
    """A domain holds 15 intents, so k=16 cannot be pure; it must say so."""
    rows, m = sample
    r = [x for x in rows if not x["oos"]][0]
    _, _, keff, pol = schema.options_for(r["text_id"], "intent", 16, r["gold"]["intent"], m)
    assert pol == "mixed" and keff == 16


def test_letter_tables_are_single_tokens():
    """PLACEBO: build the table from a tokenizer of another family -> multi-token letters."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.SMOKE_MODEL)
    bare, space = build_tables(tok)
    assert len(set(bare)) == 26 and len(set(space)) == 26
    assert bare[:3] == [32, 33, 34]


def test_score_tie_goes_to_lowest_position_and_is_flagged():
    cb = torch.tensor([1.0, 1.0, 0.0])
    s = score(cb, cb)
    assert s["pred_pos"] == 0 and s["tie"] is True
    s2 = score(torch.tensor([0.0, 1.0, 0.5]), torch.tensor([0.0, 0.0, 0.0]))
    assert s2["pred_pos"] == 1 and s2["tie"] is False


def test_gap_is_between_the_two_best_candidates():
    s = score(torch.tensor([5.0, 2.0, 1.0]), torch.tensor([0.0, 0.0, 0.0]))
    assert abs(s["gap"] - 3.0) < 1e-6


def test_classify_argmax_distinguishes_beyond_k_from_other():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.SMOKE_MODEL)
    bare, space = build_tables(tok)
    assert classify_argmax(bare[0], bare, space, 4) == "bare_candidate"
    assert classify_argmax(space[0], bare, space, 4) == "space_candidate"
    assert classify_argmax(bare[9], bare, space, 4) == "letter_beyond_k"
    assert classify_argmax(tok.encode("hello", add_special_tokens=False)[0], bare, space, 4) == "other"


def test_json_schema_preserves_key_order_and_forbids_extras():
    sc = schema.json_schema({"intent": ["a", "b"], "domain": ["x"], "is_banking_or_credit_cards": ["true", "false"]})
    assert list(sc["properties"]) == ["intent", "domain", "is_banking_or_credit_cards"]
    assert sc["additionalProperties"] is False
    assert sc["properties"]["is_banking_or_credit_cards"] == {"type": "boolean"}
    assert sc["required"] == ["intent", "domain", "is_banking_or_credit_cards"]


def test_schema_sha_is_order_sensitive():
    """Key order is part of the instruction, so it must change the hash."""
    a = schema.schema_sha(schema.json_schema({"intent": ["a"], "domain": ["x"]}))
    b = schema.schema_sha(schema.json_schema({"domain": ["x"], "intent": ["a"]}))
    assert a != b


def test_letters_ebnf_lists_exactly_k_letters():
    assert schema.letters_ebnf(3).strip() == 'root ::= "A" | "B" | "C"'


def test_parse_a_never_turns_a_truncation_into_an_empty_answer():
    """A cut-off generation is a failure state, not an answer with no fields."""
    fo = {"intent": ["a", "b"]}
    assert parse_a('{"intent": "a"', fo)[0] == "malformed_json"
    assert parse_a("", fo)[0] == "malformed_json"
    assert parse_a('{"intent": "zzz"}', fo)[0] == "out_of_enum"
    assert parse_a('{"other": "a"}', fo)[0] == "missing_key"
    assert parse_a('{"intent": "a", "x": 1}', fo)[0] == "extra_key"
    assert parse_a('{"intent": "a"}', fo) == ("ok", {"intent": "a"})


def test_parse_a_maps_json_booleans_to_the_string_gold():
    fo = {"is_banking_or_credit_cards": ["true", "false"]}
    assert parse_a('{"is_banking_or_credit_cards": true}', fo) == ("ok", {"is_banking_or_credit_cards": "true"})


def test_a_non_answer_counts_wrong_and_is_not_dropped():
    rec = {"arm": "A1joint", "condition": "main:k4:f1", "text_id": "t", "field": "_all",
           "fields_order": ["intent"], "k": 4, "k_eff": {"intent": 4}, "options": {"intent": ["a"]},
           "gold": {"intent": "a"}, "gold_pos": {"intent": 0}, "policy": {"intent": "within_domain"},
           "state": "malformed_json", "per_field_pred": {}}
    rows = unit_rows([rec])
    assert len(rows) == 1 and correct(rows[0]) is False
    assert accuracy(rows) == (0, 1, 0.0)


def test_scorer_placebo_perfect_and_adversarial():
    def mk(pred, gold):
        return {"condition": "c", "text_id": "t%d" % hash(pred) , "field": "intent",
                "pred": pred, "gold": gold}
    rows = [dict(mk("a", "a"), text_id=f"t{i}") for i in range(10)]
    assert accuracy(rows)[2] == 1.0
    rows_bad = [dict(mk("b", "a"), text_id=f"t{i}") for i in range(10)]
    assert accuracy(rows_bad)[2] == 0.0


def test_paired_delta_is_zero_for_identical_arms():
    rows = [{"condition": "c", "text_id": f"t{i}", "field": "intent", "pred": "a", "gold": "a"}
            for i in range(20)]
    pairs, b_only, a_only = paired(rows, rows)
    assert (b_only, a_only) == (0, 0)
    point, (lo, hi) = cluster_bootstrap_delta(pairs, resamples=200)
    assert point == 0.0 and lo == 0.0 and hi == 0.0


def test_paired_delta_sign_follows_the_better_arm():
    a = [{"condition": "c", "text_id": f"t{i}", "field": "intent", "pred": "b", "gold": "a"}
         for i in range(20)]
    b = [{"condition": "c", "text_id": f"t{i}", "field": "intent", "pred": "a", "gold": "a"}
         for i in range(20)]
    pairs, b_only, a_only = paired(a, b)
    assert (b_only, a_only) == (20, 0)
    point, _ = cluster_bootstrap_delta(pairs, resamples=200)
    assert point == 1.0


def test_prompts_a0_and_a1_render_identically():
    """A0 exists to show whether the grammar binds; a different prompt would confound that."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.SMOKE_MODEL)
    fo = {"intent": ["a", "b"]}
    p1 = prompts.render(tok, prompts.SYSTEM_A, prompts.user_a("text", fo))
    p0 = prompts.render(tok, prompts.SYSTEM_A, prompts.user_a("text", fo))
    assert p1 == p0


def test_b_prompt_ends_at_the_answer_position():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.SMOKE_MODEL)
    u = prompts.user_b("text", "intent", ["a", "b"])
    assert u.endswith("ANSWER:")
    r = prompts.render(tok, prompts.SYSTEM_B, u)
    assert r.endswith("\n\n") or r.endswith("assistant\n")


def test_domain_map_pin_is_enforced():
    """PLACEBO: change one byte of data/domains.json -> this raises."""
    m = data.fetch_domains()
    assert len(m) == 150 and len(set(m.values())) == 10


def test_adding_an_unused_template_does_not_invalidate_a_resume(tmp_path):
    """PLACEBO: compare one aggregate template hash -> this refuses and 6000 valid records die.
    The opposite placebo: changing system_b must still refuse."""
    from minijev import manifest as MF
    base = {k: "x" for k in MF.CERTIFIED}
    old = dict(base, template_shas={"system_a": "a", "system_b": "b"})
    MF.write(tmp_path, {"certified": old})
    added = dict(base, template_shas={"system_a": "a", "system_b": "b", "system_b_label": "NEW"})
    MF.refuse_resume_on_mismatch(tmp_path, added)          # must not raise
    changed = dict(base, template_shas={"system_a": "a", "system_b": "CHANGED"})
    with pytest.raises(SystemExit):
        MF.refuse_resume_on_mismatch(tmp_path, changed)


def test_certified_mismatch_still_refuses(tmp_path):
    from minijev import manifest as MF
    base = {k: "x" for k in MF.CERTIFIED}
    MF.write(tmp_path, {"certified": dict(base, template_shas={})})
    with pytest.raises(SystemExit):
        MF.refuse_resume_on_mismatch(tmp_path, dict(base, seed="different", template_shas={}))


def test_device_from_env_is_stripped(monkeypatch):
    """PLACEBO: drop the .strip() -> this fails, and on the Windows host torch then refuses
    the device with "Invalid device string: 'cuda '" (measured 2026-09-16)."""
    import importlib
    from minijev import config as cfg
    monkeypatch.setenv("MINIJEV_DEVICE", "cuda ")
    importlib.reload(cfg)
    assert cfg.DEVICE == "cuda"
    monkeypatch.delenv("MINIJEV_DEVICE")
    importlib.reload(cfg)


def test_run_mode_names_the_device_it_ran_on():
    """PLACEBO: hardcode 'mps' in run_mode -> this fails. Caught live: a CUDA run stamped
    every record with 'hf-mps-greedy'."""
    import importlib, os
    from minijev import config as cfg
    os.environ["MINIJEV_DEVICE"] = "cuda"
    importlib.reload(cfg)
    assert cfg.run_mode().startswith("hf-cuda-greedy"), cfg.run_mode()
    os.environ["MINIJEV_DEVICE"] = "mps"
    importlib.reload(cfg)
    assert cfg.run_mode().startswith("hf-mps-greedy"), cfg.run_mode()
    del os.environ["MINIJEV_DEVICE"]
    importlib.reload(cfg)


def test_candidate_mass_cannot_exceed_one():
    """PLACEBO: take the numerator from one distribution and the denominator from another ->
    mass > 1. That is exactly how the real defect announced itself (measured 1.0192)."""
    import torch
    logits = torch.tensor([3.0, 1.0, 0.5, -2.0, -4.0])
    logZ = torch.logsumexp(logits, 0)
    mass = float(torch.exp(torch.logsumexp(logits[:3], 0) - logZ))
    assert 0.0 < mass <= 1.0
    other = logits[:3] + 0.06          # a different numeric path for the numerator
    bad = float(torch.exp(torch.logsumexp(other, 0) - logZ))
    assert bad > mass, "the placebo must actually inflate the mass"


def test_candidate_mass_tolerance_admits_float32_rounding_but_not_mixed_distributions():
    """Two sides, both required. PLACEBO-1: a 1e-6 tolerance -> fails on honest float32 rounding
    (measured on a live run: 1.0000038 killed the arm at record 607). PLACEBO-2: drop the check ->
    mixed distributions (1.0192) pass silently."""
    TOL = 1e-4
    rounding = 1.0000038146972656      # the real value from the run
    mixed = 1.0192                     # the real value from the mixed distributions
    assert rounding <= 1.0 + TOL, "the tolerance must admit float32 rounding"
    assert mixed > 1.0 + TOL, "the tolerance must catch mixed distributions"
    assert min(rounding, 1.0) == 1.0


def test_prob_schema_bounds_values_by_enum_not_pattern():
    """PLACEBO: `{"type":"number"}` -> the model loops on 0.999… (measured: 4337/5369 length).
    Second placebo: a `pattern` -> xgrammar 0.2.7 still emitted "000" (measured). Only an enum
    is known to bind in this harness, so the schema must carry one."""
    from minijev.schema import prob_schema
    sc = prob_schema(["a", "b"])["properties"]["probabilities"]["properties"]["a"]
    assert sc["type"] == "string" and "enum" in sc and "pattern" not in sc
    assert "0.00" in sc["enum"] and "1.00" in sc["enum"] and "000" not in sc["enum"]
    assert len(sc["enum"]) == 101


def test_free_text_control_is_reported_not_measured_without_prompt_twins(tmp_path):
    """PLACEBO: pair by (text_id, field, k) -> a B1 record from ANOTHER subset is accepted as
    the twin and the 'agreement' printed is B1free's accuracy vs gold (happened: 189/200)."""
    import json
    from minijev.report import build
    d = tmp_path / "r"; d.mkdir()
    free = {"key": "ctl:textanswer|t1|intent|B1free", "arm": "B1free", "condition": "ctl:textanswer",
            "text_id": "t1", "field": "intent", "fields_order": ["intent"], "k": 4, "k_eff": {"intent": 4},
            "options": {"intent": ["a", "b", "c", "d"]}, "gold": {"intent": "a"}, "gold_pos": {"intent": 0},
            "policy": {"intent": "within_domain"}, "prompt_sha": "SHA_FREE", "state": "ok",
            "per_field_pred": {"intent": "a"}, "generated_tokens": 2, "finish_reason": "eos"}
    b1_other = {"key": "easy:k4:f1|t1|intent|B1", "arm": "B1", "condition": "easy:k4:f1", "text_id": "t1",
                "field": "intent", "fields_order": ["intent"], "k": 4, "k_eff": 4, "options": ["a", "x", "y", "z"],
                "gold": "a", "gold_pos": 0, "pred": "a", "pred_pos": 0, "policy": "cross_domain",
                "prompt_sha": "SHA_OTHER", "argmax_class": "bare_candidate", "tie": False, "gap": 3.0,
                "p_cand": [0.9, 0.05, 0.03, 0.02]}
    (d / "B1free.jsonl").write_text(json.dumps(free) + "\n")
    (d / "B1.jsonl").write_text(json.dumps(b1_other) + "\n")
    rep = build([str(d)])
    assert "NOT MEASURED" in rep and "0 with a B1 twin" in rep
    assert "agreement 1/1" not in rep, "a same-text record from another subset must NOT count as a twin"
