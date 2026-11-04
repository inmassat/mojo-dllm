from std.math import exp, log
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.diffusion.sampler import (
    GenConfig,
    num_transfer_tokens,
    select_top,
    argmax_and_prob,
    generate,
    Rng,
    _confidence,
    _confidence_from_stats,
    _greedy_rows,
    StepObserver,
    generate_observed,
)
from mojo_dllm.models.transformer import DiffusionLM, RowStats
from mojo_dllm.sys.mem import Floats

comptime FIX = "tests/fixtures/"


def test_transfer_schedule_matches_generate_py() raises:
    # LLaDA: base = m // steps, the first (m % steps) steps take one more.
    var s = num_transfer_tokens(32, 8)
    assert_equal(len(s), 8)
    for v in s:
        assert_equal(v, 4)
    var r = num_transfer_tokens(10, 4)
    assert_equal(r[0], 3)
    assert_equal(r[1], 3)
    assert_equal(r[2], 2)
    assert_equal(r[3], 2)
    var total = 0
    for v in num_transfer_tokens(31, 7):
        total += v
    assert_equal(total, 31)


def test_select_top_is_stable_on_ties() raises:
    var conf: List[Float64] = [0.5, 0.9, 0.5, 0.9, 0.1]
    var sel = select_top(conf, 3)
    assert_equal(len(sel), 3)
    assert_equal(sel[0], 1)
    assert_equal(sel[1], 3)
    assert_equal(sel[2], 0)


def test_select_top_caps_at_candidates() raises:
    var conf: List[Float64] = [0.2, 0.3]
    assert_equal(len(select_top(conf, 5)), 2)


def test_argmax_and_prob_in_f64() raises:
    var row = Floats(4)
    row[0] = 1.0
    row[1] = 3.0
    row[2] = 2.0
    row[3] = 3.0
    var r = argmax_and_prob(row.ptr(), 4)
    assert_equal(r[0], 1)  # first maximum wins
    var z = exp(Float64(-2.0)) + 1.0 + exp(Float64(-1.0)) + 1.0
    assert_true(abs(r[1] - 1.0 / z) < 1e-12, "prob " + String(r[1]))


def test_rng_is_deterministic() raises:
    var a = Rng(42)
    var b = Rng(42)
    for _ in range(10):
        assert_equal(a.next_u64(), b.next_u64())
    var r7 = Rng(7)
    var u = r7.uniform()
    assert_true(u > 0.0 and u < 1.0, "uniform in (0,1)")


def _tiny_cfg() -> GenConfig:
    var c = GenConfig()
    c.gen_length = 8
    c.block_length = 4
    c.steps = 4
    c.mask_id = 511
    return c^


def test_generation_resolves_every_mask() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var out = generate(m, prompt, _tiny_cfg())
    assert_equal(len(out.tokens), 11)
    for i in range(3):
        assert_equal(out.tokens[i], prompt[i])
    for i in range(3, 11):
        assert_true(out.tokens[i] != 511, "mask left at position " + String(i))
    assert_equal(out.forward_passes, 4)


def test_generation_is_deterministic_with_temperature() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = _tiny_cfg()
    c.temperature = 0.7
    c.seed = 1234
    var a = generate(m, prompt, c)
    var b = generate(m, prompt, c)
    for i in range(len(a.tokens)):
        assert_equal(a.tokens[i], b.tokens[i])


def test_full_logits_ablation_gives_same_tokens() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var a = generate(m, prompt, _tiny_cfg())
    var c = _tiny_cfg()
    c.full_logits = True
    var b = generate(m, prompt, c)
    for i in range(len(a.tokens)):
        assert_equal(a.tokens[i], b.tokens[i])


def test_sampler_threads_do_not_change_tokens() raises:
    # The per-row confidence pass runs rows in parallel. Each row's sums
    # keep their order, so the thread count must not move a single token.
    for path in ["tiny-llada.gguf", "tiny-dream.gguf"]:
        var m = DiffusionLM(FIX + path, threads=2, max_tokens=16)
        var prompt: List[Int] = [5, 17, 300]
        var mask = 510 if path == "tiny-dream.gguf" else 511
        var one = _tiny_cfg()
        one.threads = 1
        one.mask_id = mask
        var four = _tiny_cfg()
        four.threads = 4
        four.mask_id = mask
        var a = generate(m, prompt, one)
        var b = generate(m, prompt, four)
        assert_equal(len(a.tokens), len(b.tokens))
        for i in range(len(a.tokens)):
            assert_equal(
                a.tokens[i], b.tokens[i], path + " position " + String(i)
            )


def test_greedy_rows_match_the_sequential_scorers() raises:
    var vocab = 700
    var n = 9
    var logits = Floats(n * vocab)
    var rng = Rng(3)
    for i in range(n * vocab):
        logits[i] = Float32(rng.uniform() * 12.0 - 6.0)
    # Score rows out of order, as full-logits mode does.
    var row_of: List[Int] = [8, 0, 3, 5, 1, 7, 2, 6, 4]
    for algorithm in ["llada", "entropy", "maskgit_plus", "topk_margin"]:
        var got = _greedy_rows(logits.ptr(), row_of, vocab, algorithm, 4)
        for j in range(n):
            var row = logits.ptr().unsafe_offset(row_of[j] * vocab)
            var want: Tuple[Int, Float64]
            if algorithm == "llada":
                want = argmax_and_prob(row, vocab)
            else:
                want = _confidence(row, vocab, algorithm)
            assert_equal(
                got[0][j], want[0], algorithm + " token, row " + String(j)
            )
            assert_equal(
                got[1][j], want[1], algorithm + " confidence, row " + String(j)
            )


def _visible(tokens: List[Int], P: Int, stop: Int) -> List[Int]:
    """The generated tokens a reader sees: up to the first stop token."""
    var out = List[Int]()
    for i in range(P, len(tokens)):
        if tokens[i] == stop:
            break
        out.append(tokens[i])
    return out^


def _check_early_stop(path: String, mask: Int, block: Int) raises:
    var m = DiffusionLM(FIX + path, threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var P = len(prompt)
    var c = GenConfig()
    c.gen_length = 16
    c.block_length = block
    c.steps = 16
    c.mask_id = mask
    c.early_stop = False
    var full = generate(m, prompt, c)
    var saved_any = False
    # Each token of the full answer in turn plays the end-of-text token.
    for i in range(P, len(full.tokens)):
        var stop = full.tokens[i]
        var e = c.copy()
        e.early_stop = True
        e.stop_ids = [stop]
        var early = generate(m, prompt, e)
        var want = _visible(full.tokens, P, stop)
        var got = _visible(early.tokens, P, stop)
        assert_equal(
            len(got), len(want), path + " visible length, stop " + String(stop)
        )
        for j in range(len(want)):
            assert_equal(got[j], want[j], path + " visible token " + String(j))
        assert_true(
            early.forward_passes <= full.forward_passes,
            "early stop ran more passes",
        )
        if early.forward_passes < full.forward_passes:
            saved_any = True
    assert_true(
        saved_any,
        path + ": no stop token ever saved a pass, so the check is vacuous",
    )


def test_early_stop_keeps_the_visible_answer_llada() raises:
    _check_early_stop("tiny-llada.gguf", 511, 4)


def test_early_stop_keeps_the_visible_answer_dream() raises:
    _check_early_stop("tiny-dream.gguf", 510, 16)


def _check_stop_report(path: String, mask: Int, block: Int) raises:
    # `--json` reports steps_taken and stop_reason for every model: the passes
    # run, and why the loop ended ("answer_fixed" exactly when it stopped early).
    var m = DiffusionLM(FIX + path, threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var c = GenConfig()
    c.gen_length = 16
    c.block_length = block
    c.steps = 16
    c.mask_id = mask
    c.early_stop = False
    var full = generate(m, prompt, c)
    assert_equal(full.steps_taken, full.forward_passes)
    assert_equal(full.stop_reason, "complete", path)
    assert_true(not full.stopped_early)
    var fixed = 0
    for i in range(len(prompt), len(full.tokens)):
        var e = c.copy()
        e.early_stop = True
        e.stop_ids = [full.tokens[i]]
        var early = generate(m, prompt, e)
        assert_equal(early.steps_taken, early.forward_passes)
        if early.stopped_early:
            fixed += 1
            assert_equal(early.stop_reason, "answer_fixed", path)
            break
        assert_equal(early.stop_reason, "complete", path)
    assert_true(
        fixed > 0, path + ": no run stopped early, so the check is vacuous"
    )


def test_llada_json_members_report_steps_and_reason() raises:
    _check_stop_report("tiny-llada.gguf", 511, 4)


def test_dream_json_members_report_steps_and_reason() raises:
    _check_stop_report("tiny-dream.gguf", 510, 16)


def _threshold_cfg(threshold: Float64) -> GenConfig:
    var c = GenConfig()
    c.gen_length = 16
    c.block_length = 8
    c.steps = 16
    c.mask_id = 511
    c.early_stop = False
    c.threshold = threshold
    return c^


def test_threshold_off_or_unreachable_changes_nothing() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var base = generate(m, prompt, _threshold_cfg(0.0))
    var never = generate(m, prompt, _threshold_cfg(1.5))
    assert_equal(base.forward_passes, 16)
    assert_equal(never.forward_passes, base.forward_passes)
    for i in range(len(base.tokens)):
        assert_equal(never.tokens[i], base.tokens[i], "position " + String(i))


def test_threshold_zero_fills_each_block_in_one_step() raises:
    # Every candidate clears a threshold of 0, so each block is one pass.
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var all_in = generate(m, prompt, _threshold_cfg(1e-300))
    assert_equal(all_in.forward_passes, 2)
    for i in range(3, len(all_in.tokens)):
        assert_true(all_in.tokens[i] != 511, "mask left at " + String(i))


def test_threshold_never_takes_more_passes() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=24)
    var prompt: List[Int] = [5, 17, 300]
    var base = generate(m, prompt, _threshold_cfg(0.0))
    for tau in [0.9, 0.5, 0.1, 0.01]:
        var r = generate(m, prompt, _threshold_cfg(tau))
        assert_true(
            r.forward_passes <= base.forward_passes, "threshold " + String(tau)
        )
        for i in range(3, len(r.tokens)):
            assert_true(r.tokens[i] != 511, "mask left at " + String(i))


def test_dual_cache_with_one_step_per_block_is_exact() raises:
    # One step per block makes every pass a refresh, so the cached run must
    # match the uncached one token for token.
    var prompt: List[Int] = [5, 17, 300]
    var c = _threshold_cfg(0.0)
    c.steps = 2
    var plain = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=24)
    var a = generate(plain, prompt, c)
    var cached = DiffusionLM(
        FIX + "tiny-llada.gguf", threads=2, max_tokens=24, kv_cache=True
    )
    c.dual_cache = True
    var b = generate(cached, prompt, c)
    assert_equal(a.forward_passes, b.forward_passes)
    for i in range(len(a.tokens)):
        assert_equal(a.tokens[i], b.tokens[i], "position " + String(i))


def test_dual_cache_resolves_every_mask() raises:
    var prompt: List[Int] = [5, 17, 300]
    var c = _threshold_cfg(0.0)
    c.dual_cache = True
    var m = DiffusionLM(
        FIX + "tiny-llada.gguf", threads=2, max_tokens=24, kv_cache=True
    )
    var r = generate(m, prompt, c)
    assert_equal(r.forward_passes, 16)
    for i in range(3, len(r.tokens)):
        assert_true(r.tokens[i] != 511, "mask left at " + String(i))


def test_rejects_bad_block_geometry() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_cfg()
    c.block_length = 3
    with assert_raises(contains="multiple of the block length"):
        _ = generate(m, prompt, c)
    c = _tiny_cfg()
    c.steps = 3
    with assert_raises(contains="multiple of the number of blocks"):
        _ = generate(m, prompt, c)


def test_dream_confidences() raises:
    var row = Floats(3)
    row[0] = 0.0
    row[1] = 0.0
    row[2] = 0.0
    var e = _confidence(row.ptr(), 3, "entropy")
    assert_equal(e[0], 0)
    # uniform over 3: sum p log p = log(1/3)
    assert_true(
        abs(e[1] - log(Float64(1.0) / 3.0)) < 1e-9,
        "neg entropy " + String(e[1]),
    )
    row[2] = 2.0
    var m = _confidence(row.ptr(), 3, "maskgit_plus")
    assert_equal(m[0], 2)
    var z = 2.0 + exp(Float64(2.0))
    assert_true(abs(m[1] - exp(Float64(2.0)) / z) < 1e-12, "p(x0)")
    var k = _confidence(row.ptr(), 3, "topk_margin")
    assert_true(abs(k[1] - (exp(Float64(2.0)) - 1.0) / z) < 1e-12, "margin")


def test_dream_generation_resolves_every_mask() raises:
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = GenConfig()
    c.gen_length = 8
    c.steps = 4
    c.mask_id = 510
    var out = generate(m, prompt, c)
    assert_equal(len(out.tokens), 11)
    for i in range(3, 11):
        assert_true(out.tokens[i] != 510, "mask left at position " + String(i))
    assert_true(out.forward_passes <= 4, "at most one pass per step")


struct Recorder(StepObserver):
    var steps: Int
    var committed: Int
    var last_total: Int

    def __init__(out self):
        self.steps = 0
        self.committed = 0
        self.last_total = 0

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        self.steps += 1
        self.committed += len(committed)
        self.last_total = total


def test_observer_sees_every_step_and_commit() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var rec = Recorder()
    var out = generate_observed(m, prompt, _tiny_cfg(), rec)
    assert_equal(rec.steps, out.forward_passes)
    assert_equal(rec.committed, 8)
    assert_equal(rec.last_total, 4)


def test_observer_sees_dream_commits() raises:
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = GenConfig()
    c.gen_length = 8
    c.steps = 4
    c.mask_id = 510
    var rec = Recorder()
    _ = generate_observed(m, prompt, c, rec)
    assert_equal(rec.committed, 8)


def test_confidence_from_row_statistics_matches_the_full_row() raises:
    # The device kernel hands the sampler four numbers per row; the sampler
    # must score the row from them exactly as the loops over the whole row do.
    var vocab = 300
    var row = Floats(vocab)
    var s = UInt64(12345)
    for v in range(vocab):
        s = s * 6364136223846793005 + 1442695040888963407
        row[v] = Float32(Int((s >> 33) % 2001) - 1000) / 125.0
    row[40] = 9.0
    row[200] = 9.0  # the first of two equal maxima wins
    var best = 0
    for v in range(1, vocab):
        if row[v] > row[best]:
            best = v
    var st = RowStats()
    var m = Float64(row[best])
    var z = Float64(0)
    var a = Float64(0)
    var second = Float64(-1e300)
    for v in range(vocab):
        var e = exp(Float64(row[v]) - m)
        z += e
        a += e * (Float64(row[v]) - m)
        if v != best and Float64(row[v]) > second:
            second = Float64(row[v])
    st.tok.append(best)
    st.mx.append(m)
    st.z.append(z)
    st.second.append(second)
    st.weighted.append(a)
    assert_equal(best, 40)
    var p = argmax_and_prob(row.ptr(), vocab)
    assert_true(abs(_confidence_from_stats(st, 0, "llada") - p[1]) < 1e-12)
    for algo in ["maskgit_plus", "topk_margin", "entropy"]:
        var w = _confidence(row.ptr(), vocab, algo)
        var got = _confidence_from_stats(st, 0, algo)
        assert_true(
            abs(got - w[1]) < 1e-9,
            algo + ": " + String(got) + " against " + String(w[1]),
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
