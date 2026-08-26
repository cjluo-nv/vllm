# Soft (budget-gated) stop conditions

`soft_stop` and `soft_stop_token_ids` are stop conditions that stay **inert**
until a token budget has been spent, and behave exactly like `stop` /
`stop_token_ids` afterwards.

They exist for workloads that need to cut a generation at a **semantic
boundary at or past a budget** — for example online chain-of-thought
compression, which lets a model reason for N tokens and then summarises and
drops the reasoning. Cutting at exactly N tokens lands mid-sentence or
mid-derivation; cutting at the first `\n\n` past N does not.

## Parameters

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `soft_stop` | `str \| list[str] \| None` | `None` | Stop strings that only apply past the budget. |
| `soft_stop_token_ids` | `list[int] \| None` | `None` | Stop token ids that only apply past the budget. |
| `soft_stop_min_tokens` | `int` | `0` | The budget, in generated output tokens. |

Semantics, precisely:

- A soft condition can terminate a request only once **more than**
  `soft_stop_min_tokens` output tokens have been generated. This mirrors
  `min_tokens`, which likewise opens on the token *after* the budget.
- Soft stop strings that occur **below** the budget are ignored entirely. The
  text is kept, generation continues, and only occurrences that *complete* at
  or past the budget terminate the request.
- `min_tokens` remains a floor for everything: the effective soft gate is
  `max(min_tokens, soft_stop_min_tokens)`.
- `stop`, `stop_token_ids`, EOS and `min_tokens` are **completely unaffected**.
  A request that legitimately finishes before the budget still finishes there.
- When a soft condition fires, `finish_reason` is `"stop"` and `stop_reason` is
  the matched string (for `soft_stop`) or the matched token id (for
  `soft_stop_token_ids`), so callers can tell which condition ended the request.
- If a regular stop string and a soft stop string match in the same decode
  step, the regular one wins.
- The matched soft stop string is stripped from the output text unless
  `include_stop_str_in_output=True`, exactly like `stop`.

## Offline usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")

params = SamplingParams(
    max_tokens=4096,
    # Reason for at least 512 tokens, then stop at the next paragraph break.
    soft_stop=["\n\n"],
    soft_stop_min_tokens=512,
)

out = llm.generate("Prove that sqrt(2) is irrational.", params)[0].outputs[0]
print(out.finish_reason, repr(out.stop_reason), len(out.token_ids))
```

## Online (OpenAI-compatible) usage

All three parameters are accepted by `/v1/completions` and
`/v1/chat/completions` as vLLM extra sampling parameters:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
        "max_tokens": 4096,
        "temperature": 0,
        "soft_stop": ["\n\n"],
        "soft_stop_min_tokens": 512
      }'
```

## Why not `min_tokens`, and why not a logits processor

`min_tokens` gates **all** stop conditions at once. `SamplingParams` folds
`stop_token_ids` and the engine folds `eos_token_id` into a single
`all_stop_token_ids` set, `MinTokensLogitsProcessor` masks that whole set to
`-inf` below the budget, and the detokenizer gates stop **strings** on the same
threshold. So `min_tokens=512` combined with `stop=["\n\n"]` would also
suppress `</think>` and EOS for 512 tokens, padding out every response that
legitimately finishes early.

`soft_stop_token_ids` is therefore deliberately **not** added to
`all_stop_token_ids`. Adding it would reintroduce exactly that bug.

A custom logits processor cannot express this either. `\n\n` occurs constantly
in normal prose, so the model must be free to *emit* it and we must only
*terminate* on it past the budget. That is a stop-condition gate, not a logit
mask, and vLLM has no plugin hook for stop conditions.

## Loading this feature into an official vLLM Docker image without rebuilding

The feature touches **pure Python only** — no kernels, no compiled extensions,
no build-system changes:

```
vllm/sampling_params.py
vllm/v1/engine/detokenizer.py
vllm/v1/core/sched/utils.py
vllm/v1/engine/async_llm.py
vllm/entrypoints/openai/completion/protocol.py
vllm/entrypoints/openai/chat_completion/protocol.py
```

So it can be overlaid onto a released `vllm/vllm-openai` image at container
start. No `docker build`, no wheel, no recompilation.

This was verified end to end: a stock `vllm/vllm-openai:v0.28.0` container with
only these six files bind-mounted over `site-packages` reports
`vllm.__version__ == "0.28.0"`, exposes all three parameters on
`SamplingParams`, and enforces the budget correctly on an H200.

Note that `site-packages` is root-owned in the official images, so a container
running as a non-root user **cannot** simply copy files in — the bind-mount is
not just convenient, it is usually the only option that does not require root.

### 1. Locate `site-packages/vllm` inside the image

Do not guess the path — it moves between images (`dist-packages` vs
`site-packages`, sometimes a venv). Ask Python:

```bash
docker run --rm --entrypoint python3 vllm/vllm-openai:v0.28.0 \
  -c 'import vllm, os; print(os.path.dirname(vllm.__file__))'
# /usr/local/lib/python3.12/dist-packages/vllm
```

Export it once and reuse it:

```bash
VLLM_PKG=$(docker run --rm --entrypoint python3 vllm/vllm-openai:v0.28.0 \
  -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
```

### 2. Bind-mount approach (recommended)

Keep the six patched files in a directory that mirrors the package layout:

```
/srv/soft-stop-overlay/
└── vllm/
    ├── sampling_params.py
    ├── entrypoints/openai/completion/protocol.py
    ├── entrypoints/openai/chat_completion/protocol.py
    └── v1/
        ├── core/sched/utils.py
        └── engine/{async_llm.py,detokenizer.py}
```

Mount each file read-only over its installed counterpart:

```bash
OVERLAY=/srv/soft-stop-overlay
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm

docker run --rm --gpus all -p 8000:8000 \
  -v $OVERLAY/vllm/sampling_params.py:$VLLM_PKG/sampling_params.py:ro \
  -v $OVERLAY/vllm/v1/engine/detokenizer.py:$VLLM_PKG/v1/engine/detokenizer.py:ro \
  -v $OVERLAY/vllm/v1/engine/async_llm.py:$VLLM_PKG/v1/engine/async_llm.py:ro \
  -v $OVERLAY/vllm/v1/core/sched/utils.py:$VLLM_PKG/v1/core/sched/utils.py:ro \
  -v $OVERLAY/vllm/entrypoints/openai/completion/protocol.py:$VLLM_PKG/entrypoints/openai/completion/protocol.py:ro \
  -v $OVERLAY/vllm/entrypoints/openai/chat_completion/protocol.py:$VLLM_PKG/entrypoints/openai/chat_completion/protocol.py:ro \
  vllm/vllm-openai:v0.28.0 --model Qwen/Qwen3-0.6B
```

Mount **individual files**, not the `vllm/` directory — mounting the directory
would shadow the compiled `.so` extensions and the rest of the package.

Stale `__pycache__` is not a concern: CPython compares the source mtime and
size recorded in the `.pyc` against the file it is importing, and a bind-mounted
file has different values, so the cached bytecode is discarded and the mounted
source is recompiled. If you have a hardened image where `__pycache__` was
pre-built with `--invalidation-mode unchecked-hash`, set `PYTHONDONTWRITEBYTECODE=1`
and delete the relevant `__pycache__` directories in the entrypoint instead.

Same thing under Slurm/Pyxis:

```bash
srun --gres=gpu:1 \
  --container-image=vllm/vllm-openai:v0.28.0 \
  --container-mounts=\
/srv/soft-stop-overlay/vllm/sampling_params.py:/usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py,\
/srv/soft-stop-overlay/vllm/v1/engine/detokenizer.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py,\
/srv/soft-stop-overlay/vllm/v1/engine/async_llm.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py,\
/srv/soft-stop-overlay/vllm/v1/core/sched/utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/utils.py,\
/srv/soft-stop-overlay/vllm/entrypoints/openai/completion/protocol.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/protocol.py,\
/srv/soft-stop-overlay/vllm/entrypoints/openai/chat_completion/protocol.py:/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py \
  vllm serve Qwen/Qwen3-0.6B
```

### 3. Patch-file approach

Produce the patch once, from a checkout at the tag that matches your image:

```bash
git diff v0.28.0..soft-stop-conditions -- vllm/ > soft-stop.patch
```

Apply it at container start (needs a writable overlay on `site-packages`, which
is the default for `docker run` without `--read-only`):

```bash
#!/usr/bin/env bash
# entrypoint-soft-stop.sh
set -euo pipefail

VLLM_PKG=$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')

# -p2 strips "a/vllm/", so paths resolve relative to the package directory.
patch -d "$VLLM_PKG" -p2 --forward --batch < /patches/soft-stop.patch

python3 - <<'PY'
from vllm import SamplingParams
assert "soft_stop" in SamplingParams.__struct_fields__, "soft-stop patch not active"
print("soft-stop patch active")
PY

exec vllm serve "$@"
```

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /srv/patches:/patches:ro \
  -v /srv/entrypoint-soft-stop.sh:/entrypoint-soft-stop.sh:ro \
  --entrypoint /entrypoint-soft-stop.sh \
  vllm/vllm-openai:v0.28.0 Qwen/Qwen3-0.6B
```

Prefer the bind-mount when you control the file set (it is atomic and
idempotent); prefer the patch file when you want the change to survive across
adjacent vLLM versions, since `patch` can absorb small context drift that a
whole-file overlay cannot.

### 4. Version pinning — the part that bites

Both approaches are pinned to a specific vLLM version. A whole-file overlay
from a different version silently replaces working code with code written
against different internals. A patch file may fail to apply, or — worse —
apply with fuzz at shifted line numbers.

Three defences, in order:

**a. Pin the image by digest, not by tag.** Tags such as `latest` and even
`v0.28.0` can be re-pushed.

```bash
docker image inspect --format '{{index .RepoDigests 0}}' vllm/vllm-openai:v0.28.0
# vllm/vllm-openai@sha256:...
```

**b. Assert the version before patching.** Fail loudly rather than serving
something half-patched:

```bash
EXPECTED=0.28.0
ACTUAL=$(python3 -c 'import vllm; print(vllm.__version__)')
[ "$ACTUAL" = "$EXPECTED" ] || {
  echo "vLLM version mismatch: image has $ACTUAL, overlay built for $EXPECTED" >&2
  exit 1
}
```

For a patch file, also refuse fuzzy application — `patch --fuzz=0` (or
`git apply`, which never fuzzes) turns "applied at shifted line numbers" from a
silent success into an error.

**c. Assert the feature is actually live at runtime.** This matters more than
it looks: the OpenAI-compatible request models use `extra="allow"`, so an
**unpatched** server accepts `soft_stop` and silently ignores it. Your requests
succeed, the budget is never enforced, and nothing in the logs says so.

In-process check:

```bash
python3 -c 'from vllm import SamplingParams; assert "soft_stop" in SamplingParams.__struct_fields__'
```

Over HTTP, against the already-running server — `soft_stop_min_tokens >
max_tokens` is rejected by a patched server and ignored by an unpatched one:

```bash
code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"hi","max_tokens":4,"soft_stop_min_tokens":9999}')
[ "$code" = "400" ] && echo "soft-stop active" || echo "SOFT-STOP NOT ACTIVE (got $code)"
```

### 5. When this trick stops working

The overlay works **only** because every changed file is pure Python that is
imported at runtime. It does not extend to:

- CUDA/HIP kernels or anything else in `csrc/` — those live in compiled
  `.so` extensions (`vllm._C`, `vllm._moe_C`, ...) inside the image;
- changes to `torch.compile` / CUDA-graph-captured code paths whose artifacts
  are baked into the image's compile cache;
- new dependencies, entry points, or `pyproject.toml` metadata.

Any of those require a real image rebuild (`docker build -f docker/Dockerfile .`).
Before shipping an overlay, confirm the change is Python-only:

```bash
git diff --name-only v0.28.0..soft-stop-conditions | grep -v '\.py$' && echo "NOT python-only"
```
