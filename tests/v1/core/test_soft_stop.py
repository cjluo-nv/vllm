# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for budget-gated ("soft") stop conditions.

`soft_stop` / `soft_stop_token_ids` are inert until `soft_stop_min_tokens`
output tokens have been generated, and must not disturb `stop`,
`stop_token_ids`, EOS or `min_tokens` in any way.
"""

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import Request, RequestStatus

pytestmark = pytest.mark.cpu_test

SOFT_TOK = 999
HARD_TOK = 888
EOS_TOK = 2


def _make_request(params: SamplingParams) -> Request:
    return Request(
        request_id="test",
        prompt_token_ids=[1, 2, 3],
        sampling_params=params,
        pooling_params=None,
    )


# ---------------------------------------------------------------------------
# SamplingParams: normalization + validation
# ---------------------------------------------------------------------------


def test_defaults_are_inert():
    params = SamplingParams()
    assert params.soft_stop == []
    assert params.soft_stop_token_ids == []
    assert params.soft_stop_min_tokens == 0


def test_soft_stop_string_is_normalized_to_list():
    params = SamplingParams(soft_stop="\n\n")
    assert params.soft_stop == ["\n\n"]


def test_soft_stop_token_ids_are_deduplicated():
    params = SamplingParams(soft_stop_token_ids=[5, 5, 7])
    assert params.soft_stop_token_ids == [5, 7]


def test_soft_stop_token_ids_never_leak_into_all_stop_token_ids():
    """The critical invariant.

    `all_stop_token_ids` is what `MinTokensLogitsProcessor` masks to -inf. If
    soft stop tokens landed in there, `min_tokens` would suppress them and the
    model could not emit them below the budget -- exactly the behaviour soft
    stops exist to avoid.
    """
    params = SamplingParams(
        stop_token_ids=[HARD_TOK],
        soft_stop_token_ids=[SOFT_TOK],
        min_tokens=8,
        max_tokens=64,
    )
    assert HARD_TOK in params.all_stop_token_ids
    assert SOFT_TOK not in params.all_stop_token_ids

    # ... and still not after the engine folds in the EOS token id.
    params.update_from_generation_config({}, eos_token_id=EOS_TOK)
    assert EOS_TOK in params.all_stop_token_ids
    assert SOFT_TOK not in params.all_stop_token_ids

    # ... nor after a generation_config that carries extra EOS ids.
    params.update_from_generation_config(
        {"eos_token_id": [EOS_TOK, 7]}, eos_token_id=EOS_TOK
    )
    assert SOFT_TOK not in params.all_stop_token_ids
    assert SOFT_TOK not in (params.stop_token_ids or [])


def test_negative_soft_stop_min_tokens_rejected():
    with pytest.raises(VLLMValidationError, match="soft_stop_min_tokens"):
        SamplingParams(soft_stop_min_tokens=-1)


def test_soft_stop_min_tokens_above_max_tokens_rejected():
    with pytest.raises(VLLMValidationError, match="soft_stop_min_tokens"):
        SamplingParams(soft_stop_min_tokens=100, max_tokens=10)


def test_soft_stop_min_tokens_equal_to_max_tokens_allowed():
    SamplingParams(soft_stop_min_tokens=10, max_tokens=10)


def test_soft_stop_min_tokens_allowed_when_max_tokens_is_none():
    SamplingParams(soft_stop_min_tokens=100, max_tokens=None)


def test_empty_soft_stop_string_rejected():
    with pytest.raises(VLLMValidationError, match="soft_stop cannot contain"):
        SamplingParams(soft_stop=["ok", ""])


def test_non_integer_soft_stop_token_ids_rejected():
    with pytest.raises(VLLMValidationError, match="soft_stop_token_ids"):
        SamplingParams(soft_stop_token_ids=["not-an-int"])


def test_soft_stop_requires_detokenize():
    with pytest.raises(VLLMValidationError, match="soft_stop strings"):
        SamplingParams(soft_stop=["\n\n"], detokenize=False)


def test_soft_stop_strings_extend_the_output_text_buffer():
    params = SamplingParams(stop=["ab"], soft_stop=["abcdef"])
    assert params.output_text_buffer_length == len("abcdef") - 1


def test_repr_includes_soft_fields():
    text = repr(SamplingParams(soft_stop=["\n\n"], soft_stop_min_tokens=4))
    assert "soft_stop=" in text
    assert "soft_stop_token_ids=" in text
    assert "soft_stop_min_tokens=4" in text


def test_from_optional_plumbs_soft_fields():
    params = SamplingParams.from_optional(
        soft_stop="\n\n", soft_stop_token_ids=[SOFT_TOK], soft_stop_min_tokens=3
    )
    assert params.soft_stop == ["\n\n"]
    assert params.soft_stop_token_ids == [SOFT_TOK]
    assert params.soft_stop_min_tokens == 3


# ---------------------------------------------------------------------------
# check_stop: soft stop token ids
# ---------------------------------------------------------------------------


def test_soft_stop_token_ignored_below_budget():
    params = SamplingParams(
        max_tokens=100, soft_stop_token_ids=[SOFT_TOK], soft_stop_min_tokens=5
    )
    request = _make_request(params)
    request.append_output_token_ids([10, 20, SOFT_TOK])
    assert not check_stop(request, max_model_len=1024)
    assert request.status != RequestStatus.FINISHED_STOPPED


def test_soft_stop_token_ignored_exactly_at_budget():
    """Mirrors `min_tokens`: the gate opens on the token *after* the budget."""
    params = SamplingParams(
        max_tokens=100, soft_stop_token_ids=[SOFT_TOK], soft_stop_min_tokens=3
    )
    request = _make_request(params)
    request.append_output_token_ids([10, 20, SOFT_TOK])
    assert request.num_output_tokens == 3
    assert not check_stop(request, max_model_len=1024)


def test_soft_stop_token_fires_past_budget():
    params = SamplingParams(
        max_tokens=100, soft_stop_token_ids=[SOFT_TOK], soft_stop_min_tokens=3
    )
    request = _make_request(params)
    request.append_output_token_ids([10, 20, 30, SOFT_TOK])
    assert check_stop(request, max_model_len=1024)
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.stop_reason == SOFT_TOK


def test_hard_stop_token_still_fires_below_the_soft_budget():
    """The regression `min_tokens` would have caused: a request that legitimately
    finishes inside the first chunk must still finish there."""
    params = SamplingParams(
        max_tokens=1000,
        stop_token_ids=[HARD_TOK],
        soft_stop_token_ids=[SOFT_TOK],
        soft_stop_min_tokens=1000,
    )
    request = _make_request(params)
    request.append_output_token_ids([10, HARD_TOK])
    assert check_stop(request, max_model_len=1024)
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.stop_reason == HARD_TOK


def test_eos_still_fires_below_the_soft_budget():
    params = SamplingParams(
        max_tokens=1000, soft_stop_token_ids=[SOFT_TOK], soft_stop_min_tokens=1000
    )
    params.update_from_generation_config({}, eos_token_id=EOS_TOK)
    request = _make_request(params)
    request.append_output_token_ids([10, EOS_TOK])
    assert check_stop(request, max_model_len=1024)
    assert request.status == RequestStatus.FINISHED_STOPPED


def test_min_tokens_remains_a_floor_for_soft_stop_tokens():
    params = SamplingParams(
        max_tokens=100,
        min_tokens=10,
        soft_stop_token_ids=[SOFT_TOK],
        soft_stop_min_tokens=2,
    )
    request = _make_request(params)
    request.append_output_token_ids([10, 20, SOFT_TOK])
    assert not check_stop(request, max_model_len=1024)

    request.append_output_token_ids([30] * 8 + [SOFT_TOK])
    assert request.num_output_tokens > 10
    assert check_stop(request, max_model_len=1024)
    assert request.stop_reason == SOFT_TOK


def test_max_tokens_still_caps_when_no_soft_boundary_is_found():
    params = SamplingParams(
        max_tokens=5, soft_stop_token_ids=[SOFT_TOK], soft_stop_min_tokens=2
    )
    request = _make_request(params)
    request.append_output_token_ids([10, 20, 30, 40, 50])
    assert check_stop(request, max_model_len=1024)
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_unrelated_request_unaffected():
    params = SamplingParams(max_tokens=100)
    request = _make_request(params)
    request.append_output_token_ids([10, SOFT_TOK, 20])
    assert not check_stop(request, max_model_len=1024)
