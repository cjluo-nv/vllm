# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Detokenizer-level tests for budget-gated ("soft") stop strings.

The tokens fed to the detokenizer are just the tokenization of a fixed piece
of text, so these tests are deterministic and need no model.
"""

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.detokenizer import FastIncrementalDetokenizer

# Five sentences, so "." occurs repeatedly and at increasing token offsets.
TEXT = (
    "one two three four. five six seven eight. nine ten eleven twelve. "
    "thirteen fourteen fifteen sixteen. seventeen eighteen nineteen twenty."
)
SEP = "."
MANY = 1000  # max_tokens large enough that it never caps these tests


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/opt-125m")


@pytest.fixture(scope="module")
def token_ids(tokenizer):
    return tokenizer(TEXT, add_special_tokens=False).input_ids


def _detokenizer(tokenizer, params: SamplingParams) -> FastIncrementalDetokenizer:
    request = EngineCoreRequest(
        request_id="",
        prompt_token_ids=[tokenizer.bos_token_id],
        mm_features=None,
        sampling_params=params,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    return FastIncrementalDetokenizer(tokenizer, request)


def _feed_one_by_one(tokenizer, params: SamplingParams, token_ids: list[int]):
    """Feed one token per `update()` call, as the engine does at decode time.

    Returns (stop_string, num_output_tokens, output_text).
    """
    detokenizer = _detokenizer(tokenizer, params)
    for token_id in token_ids:
        stop_string = detokenizer.update([token_id], False)
        if stop_string is not None:
            return stop_string, detokenizer.num_output_tokens(), detokenizer.output_text
    return None, detokenizer.num_output_tokens(), detokenizer.output_text


def _feed_at_once(tokenizer, params: SamplingParams, token_ids: list[int]):
    """Feed every token in a single `update()` call (speculative-decode style).

    The token count is not meaningful here -- every token is consumed before
    stop evaluation runs -- so only the text and the matched string are
    returned.
    """
    detokenizer = _detokenizer(tokenizer, params)
    stop_string = detokenizer.update(list(token_ids), False)
    return stop_string, detokenizer.output_text


FEEDERS = [_feed_one_by_one, _feed_at_once]


def _text_and_stop(feed, tokenizer, params, token_ids):
    """Feeder-independent view: (stop_string, output_text)."""
    result = feed(tokenizer, params, token_ids)
    return (result[0], result[-1])


def _full_text(tokenizer, token_ids) -> str:
    """The generation with no stop conditions at all."""
    return _feed_one_by_one(tokenizer, SamplingParams(max_tokens=MANY), token_ids)[2]


def _boundary_token_counts(tokenizer, token_ids) -> list[int]:
    """`num_output_tokens` at which each SEP completes, one token per step."""
    detokenizer = _detokenizer(tokenizer, SamplingParams(max_tokens=MANY))
    counts: list[int] = []
    seen = 0
    for token_id in token_ids:
        detokenizer.update([token_id], False)
        total = detokenizer.output_text.count(SEP)
        while seen < total:
            seen += 1
            counts.append(detokenizer.num_output_tokens())
    return counts


def test_fixture_has_several_boundaries(tokenizer, token_ids):
    """Guard: the rest of the file is meaningless without several separators."""
    counts = _boundary_token_counts(tokenizer, token_ids)
    assert len(counts) >= 4, counts
    assert counts == sorted(counts)


@pytest.mark.parametrize("feed", FEEDERS)
def test_soft_stop_with_zero_budget_matches_plain_stop(tokenizer, token_ids, feed):
    hard = _text_and_stop(feed, tokenizer, SamplingParams(stop=[SEP]), token_ids)
    soft = _text_and_stop(feed, tokenizer, SamplingParams(soft_stop=[SEP]), token_ids)
    assert hard == soft
    assert hard == (SEP, "one two three four")


def test_soft_stop_ignores_matches_below_the_budget(tokenizer, token_ids):
    budget = 8
    stop_string, num_tokens, text = _feed_one_by_one(
        tokenizer,
        SamplingParams(soft_stop=[SEP], soft_stop_min_tokens=budget, max_tokens=MANY),
        token_ids,
    )
    counts = _boundary_token_counts(tokenizer, token_ids)
    early = [c for c in counts if c <= budget]
    expected_cut = next(c for c in counts if c > budget)

    assert stop_string == SEP
    # Terminated at or past the budget ...
    assert num_tokens >= budget
    # ... at the FIRST boundary past it ...
    assert num_tokens == expected_cut
    # ... having walked over the earlier separators, which are still in the
    # text (proving they were ignored rather than suppressed) ...
    assert early
    assert text.count(SEP) == len(early)
    # ... and cut exactly on the boundary: the full generation continues with
    # the separator right where the kept text ends.
    assert _full_text(tokenizer, token_ids).startswith(text + SEP)


@pytest.mark.parametrize("budget", [0, 4, 8, 16, 24])
def test_cut_is_always_the_first_boundary_past_the_budget(tokenizer, token_ids, budget):
    counts = _boundary_token_counts(tokenizer, token_ids)
    expected = next((c for c in counts if c > budget), None)
    _, num_tokens, _ = _feed_one_by_one(
        tokenizer,
        SamplingParams(soft_stop=[SEP], soft_stop_min_tokens=budget, max_tokens=MANY),
        token_ids,
    )
    assert num_tokens == expected


def test_larger_budget_cuts_later(tokenizer, token_ids):
    small = _feed_one_by_one(
        tokenizer,
        SamplingParams(soft_stop=[SEP], soft_stop_min_tokens=6, max_tokens=MANY),
        token_ids,
    )
    large = _feed_one_by_one(
        tokenizer,
        SamplingParams(soft_stop=[SEP], soft_stop_min_tokens=20, max_tokens=MANY),
        token_ids,
    )
    assert small[1] < large[1]
    assert large[2].startswith(small[2])


@pytest.mark.parametrize("feed", FEEDERS)
def test_no_boundary_found(tokenizer, token_ids, feed):
    """A soft stop string that never occurs past the budget never fires."""
    stop_string, text = _text_and_stop(
        feed,
        tokenizer,
        SamplingParams(
            soft_stop=["<<<never>>>"], soft_stop_min_tokens=4, max_tokens=MANY
        ),
        token_ids,
    )
    assert stop_string is None
    assert text == _full_text(tokenizer, token_ids)


@pytest.mark.parametrize("feed", FEEDERS)
def test_hard_stop_still_wins_early(tokenizer, token_ids, feed):
    """A huge soft budget must not delay a regular stop string."""
    baseline = _text_and_stop(
        feed, tokenizer, SamplingParams(stop=[SEP], max_tokens=MANY), token_ids
    )
    with_soft = _text_and_stop(
        feed,
        tokenizer,
        SamplingParams(
            stop=[SEP],
            soft_stop=["\n\n"],
            soft_stop_min_tokens=500,
            max_tokens=MANY,
        ),
        token_ids,
    )
    assert baseline == with_soft
    assert baseline[0] == SEP


def test_min_tokens_is_a_floor_for_soft_stops(tokenizer, token_ids):
    """`min_tokens` keeps its guarantee even for soft conditions."""
    counts = _boundary_token_counts(tokenizer, token_ids)
    ungated = _feed_one_by_one(
        tokenizer,
        SamplingParams(soft_stop=[SEP], soft_stop_min_tokens=0, max_tokens=MANY),
        token_ids,
    )
    gated = _feed_one_by_one(
        tokenizer,
        SamplingParams(
            soft_stop=[SEP], soft_stop_min_tokens=0, min_tokens=12, max_tokens=MANY
        ),
        token_ids,
    )
    assert ungated[1] == counts[0]
    assert gated[1] == next(c for c in counts if c > 12)


@pytest.mark.parametrize("feed", FEEDERS)
def test_include_stop_str_in_output(tokenizer, token_ids, feed):
    excluded = _text_and_stop(
        feed,
        tokenizer,
        SamplingParams(soft_stop=[SEP], soft_stop_min_tokens=8, max_tokens=MANY),
        token_ids,
    )
    included = _text_and_stop(
        feed,
        tokenizer,
        SamplingParams(
            soft_stop=[SEP],
            soft_stop_min_tokens=8,
            max_tokens=MANY,
            include_stop_str_in_output=True,
        ),
        token_ids,
    )
    assert included[1] == excluded[1] + SEP


def test_stop_buffer_length_accounts_for_soft_stop_strings(tokenizer):
    params = SamplingParams(stop=["ab"], soft_stop=["abcdef"])
    detokenizer = _detokenizer(tokenizer, params)
    assert detokenizer.stop_buffer_length == len("abcdef") - 1


@pytest.mark.parametrize("feed", FEEDERS)
@pytest.mark.parametrize(
    "base_kwargs",
    [
        {},
        {"stop": [SEP]},
        {"stop": [SEP], "min_tokens": 5},
        {"stop": [SEP], "include_stop_str_in_output": True},
    ],
)
def test_inert_soft_params_leave_behaviour_untouched(
    tokenizer, token_ids, feed, base_kwargs
):
    """Adding soft params that can never match must change nothing."""
    kwargs = {"max_tokens": MANY, **base_kwargs}
    without = _text_and_stop(feed, tokenizer, SamplingParams(**kwargs), token_ids)
    with_inert = _text_and_stop(
        feed,
        tokenizer,
        SamplingParams(soft_stop=["<<<never>>>"], soft_stop_min_tokens=0, **kwargs),
        token_ids,
    )
    assert without == with_inert
