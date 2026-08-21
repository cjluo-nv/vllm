#!/bin/bash
# Start vLLM + the CoT-compression sidecar on one GCP-NRT node and run the
# end-to-end test.  Submit from the login node:
#   sbatch examples/cot_compression/run_gcp_nrt.sbatch
#SBATCH --account=coreai_numerics_nemotron
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --job-name=coreai_numerics_nemotron-cot.sidecar
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/chenjiel/cot-compress/logs/cot-%j.out

set -x
IMG=/lustre/fsw/portfolios/coreai/users/chenjiel/images/qwen38-x86_64-cu129.sqsh
WORK=/lustre/fsw/portfolios/coreai/users/chenjiel/cot-compress
MODEL_PATH=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_modelopt/hf-local/Qwen/Qwen3.8-27B-FP8
TP=${TP:-1}
KEEP_ALIVE=${KEEP_ALIVE:-3000}
RATIO=${RATIO:-0.3}

mkdir -p "$WORK/logs" "$WORK/out"

srun --container-image="$IMG" \
     --container-mounts=/lustre:/lustre \
     --no-container-mount-home \
     --export=ALL,MODEL_PATH="$MODEL_PATH",WORK="$WORK",TP="$TP",KEEP_ALIVE="$KEEP_ALIVE",RATIO="$RATIO",SLURM_JOB_ID="$SLURM_JOB_ID" \
     bash -lc '
set -x
cd "$WORK"
export VLLM_LOGGING_LEVEL=INFO
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=================== vLLM serve ==================="
vllm serve "$MODEL_PATH" \
  --served-model-name qwen38 \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size "$TP" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  > "$WORK/logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!

python3 - <<PY
import json, time, urllib.request
for i in range(360):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5).read()
        print("[boot] vllm healthy after", i*5, "s", flush=True); break
    except Exception as e:
        if i % 12 == 0: print("[boot] waiting for vllm", i*5, "s", repr(e)[:90], flush=True)
        time.sleep(5)
else:
    raise SystemExit("vllm never became healthy")

paths = json.load(urllib.request.urlopen("http://127.0.0.1:8000/openapi.json", timeout=30))["paths"]
need = ["/v1/chat/completions/render", "/inference/v1/generate", "/tokenize", "/detokenize"]
print("[boot] required routes:", {p: (p in paths) for p in need}, flush=True)
missing = [p for p in need if p not in paths]
if missing:
    raise SystemExit(f"missing routes: {missing}")
PY
if [ $? -ne 0 ]; then
  echo "=== vLLM FAILED TO START; tail of server log ==="
  tail -100 "$WORK/logs/vllm-$SLURM_JOB_ID.log"
  exit 1
fi

echo "=================== sidecar ==================="
export VLLM_URL=http://127.0.0.1:8000
export MODEL=qwen38
export ARM=self
export RATIO="$RATIO"
export THINK_BUDGET=4096
export ANSWER_BUDGET=1024
export TRACE_LOG="$WORK/out/traces-$SLURM_JOB_ID.jsonl"
export SIDECAR_PORT=9000
python3 "$WORK/cot_sidecar.py" > "$WORK/logs/sidecar-$SLURM_JOB_ID.log" 2>&1 &
SIDECAR_PID=$!
sleep 10
echo "--- sidecar startup log ---"
cat "$WORK/logs/sidecar-$SLURM_JOB_ID.log"

echo "=================== test ==================="
python3 "$WORK/test_sidecar.py" \
  --sidecar http://127.0.0.1:9000 \
  --vllm http://127.0.0.1:8000 \
  --model qwen38 --ratio "$RATIO" \
  2>&1 | tee "$WORK/out/test-$SLURM_JOB_ID.txt"
TEST_RC=${PIPESTATUS[0]}

echo "=================== sidecar log ==================="
tail -60 "$WORK/logs/sidecar-$SLURM_JOB_ID.log"
echo "=================== trace log ==================="
python3 - <<PY
import json
p = "$WORK/out/traces-$SLURM_JOB_ID.jsonl"
rows = [json.loads(l) for l in open(p)]
print(f"{len(rows)} records in {p}")
for r in rows:
    p1 = r["phase1"]; c = r["compress"]
    print("  arm=%-9s p1=%-6d comp=%-6d ratio=%.3f closed=%s retok_ident=%s err=%s"
          % (r["arm"], p1["n_tokens"], c["n_tokens"], c["achieved_ratio"],
             p1["closed"], c["retok_identical"], c["error"]))
PY

echo "TEST_RC=$TEST_RC"
echo "=================== keep-alive ${KEEP_ALIVE}s (node: $(hostname)) ==================="
sleep "$KEEP_ALIVE"
kill $SIDECAR_PID $VLLM_PID 2>/dev/null
exit $TEST_RC
'
