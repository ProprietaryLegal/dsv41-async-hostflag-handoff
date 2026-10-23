# DeepSeek-V4.1-Flash async host-flag handoff: DGX Spark GB10 and x86 RTX 3090 drafting

A CUDA-free host-flag handoff lets the DGX Spark engine overlap remote DSpark rounds with engine work. A separate proxy process removes the measured prefill penalty of the earlier thread proxy; bounded failure handling restores drafting after a proxy loss.

**Hugging Face:** [Card and files](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff) · **Stack overview:** [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · **Collection:** [All entries](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e)

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

## Status

2026-10-23 — production with the one-card RTX 3090 worker and the async process proxy; TP2 worker promotion is pending.

The production configuration uses capacity mode, remote Engram and the expanded KV pool. The host-flag handshake, recovery paths and supervised GB10 drills are qualified on the tested image and model; other images and GPUs require their own ordering checks.

## What it is

With a remote drafter, synchronous rounds make the engine's CPU wait for the network, which loses the overlap the local
drafter had. The async design avoids that:

1. The engine stream parks on a device flag (`cuStreamWaitValue32`).
2. A proxy does the network round.
3. The flag is released after the reply lands.

On GB10 the first version deadlocked twice:

1. **Self-test, rank 0 stuck in `cuLaunchKernel`.** The main stream was parked on the flag, then the CPU launched a kernel
   for the first time (`torch.cuda._sleep`). Under CUDA's default **lazy module loading**, a first launch loads the kernel's
   module and waits for the device. The device can't go idle while the stream is parked, so it deadlocks. This reproduces
   on an RTX 3090 ([reproduction](code/results/C/lazy-repro/repro.py)): the original self-test hangs, and passes in 0.02 s with
   `CUDA_MODULE_LOADING=EAGER` or with every kernel pre-launched.
2. **Warm-up, with `CUDA_MODULE_LOADING=EAGER`.** gdb shows:
   - the engine thread spinning in `cuModuleLoadData`, called from Triton's `loadBinary` (a first-use JIT kernel; eager
     loading does not cover explicit module loads);
   - the proxy thread blocked in `cuMemcpyHtoDAsync`, the reply copy that precedes the release.

   **Any CUDA call on the proxy side between the engine's wait and the release can deadlock against a module load**, so
   eager loading is not enough.

### CUDA-free handoff

The engine stream and the proxy meet through 32-bit counters in pinned, host-mapped memory:

- **Engine stream:** gather → D2H of the rows into pinned memory → **write `gathered = t`** (`cuStreamWriteValue32` on the
  host-mapped word) → **wait `released ≥ t`** (`cuStreamWaitValue32` on host memory) → copy the reply in.
- **Proxy:** spin on `gathered` in a small C library (`hostflag.c`: acquire/release atomics, no CUDA) → network round →
  write the reply into pinned memory → **store `released = t`**. It never calls CUDA or torch, and a test fails if it does.
- **Failures** (timeout, breaker, enqueue error, disconnect) still release the stream, again with no CUDA call. A watchdog
  releases any round older than its deadline (200 ms) and the round decodes target-only.
- **Startup self-test** exercises the mechanism with every kernel pre-launched. It refuses async, logging the reason, if
  the device can't use host pointers for stream memory operations.

The current proxy uses a shared memfd region for flags, descriptors, staged rows and replies. A native watchdog claims overdue rounds and publishes zero drafts. Only one publisher can win a round. The GPU acknowledges consumption before the reply buffer is reused. On loss, the engine reaps the old process and drains the stream before resetting flags and starting a fresh worker session.

## Why

Synchronous remote rounds block the engine CPU on the network. A CUDA-calling helper can deadlock against a module load while the engine stream waits for that helper. Moving the round outside the engine process also avoids sharing its Python lock.

## When to use it

Use it with the remote DSpark drafter when the tested stream-memory ordering holds. The mechanism is also relevant to remote fetches whose completion releases a parked GPU stream. Use the process proxy with a brief sleep between polls (the nap setting) for the measured production behavior.

## When not to use it

Do not enable it on a platform that cannot use host pointers for stream memory operations, or without validating visibility and buffer reuse. EAGER loading alone does not make a CUDA-calling proxy safe. The in-process thread proxy remains useful as a diagnostic, but its measured prefill penalty makes it unsuitable for this production configuration.

## Requirements

Two DGX Spark (GB10) nodes plus an x86 offload host with RTX 3090s; the existing remote DSpark setup; an architecture-matched hostflag library; shared-memory registration and a successful startup self-test. The proxy interpreter needs NumPy and the transport, and must not load torch. The engine needs the tested vLLM image, PyTorch and Triton. The worker and transport are separate sibling packages.

## Measured effect

All engine results below use the [EXL3 2.75 bpw build](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark), two DGX Spark GB10 nodes and an x86 RTX 3090 drafter. Step times are direct per-step timer p50 per request, not delivered-token rates inferred from acceptance.

### Earlier deadlock and thread-proxy history (2026-10-22–2026-10-23)

On an RTX 3090 (mock worker) the old design deadlocked at the first fresh `cuModuleLoadData` during a parked round. The new
one completed 30 of 30 such rounds and a 10,000-round faulted soak with 19,082 drafts bit-exact.

**The in-process version cost time to first token on GB10.** It ran the hand-off proxy as a thread inside the engine
process. Measured against synchronous rounds on the same setup (2–3 repeats per workload):

- decode was −3.3 to −4.5 % per round against synchronous rounds;
- but time to first token was **+4 % at 32K and +5–8 % at 150K**;
- 31 rounds per benchmark set hit the watchdog.

We ruled out the two obvious suspects, each by its own run:

- the interpreter switch interval: no change at the default 5 ms;
- `CUDA_MODULE_LOADING=EAGER`: synchronous + EAGER had normal time to first token. EAGER also makes every serve start take
  1,456 s instead of 364 s.

**What fixed it: the proxy as a separate, torch-free process.**

- The two sides share one `memfd` region: hand-off words, staging rows and replies. It is registered with the GPU through
  `cuMemHostRegister` and works on GB10's coherent memory.
- The proxy process imports no torch. It is 32 MB resident, with all 20 cores allowed. It uses the same `hostflag.c`
  waits, with a short spin and then a nap.

The controlled comparison on 2026-10-23 used capacity mode with remote Engram and expanded KV, cold prefixes, 1,500-token budgets and temperature 1. Short and 32K single-stream workloads had three repeats; 150K had two; the concurrent workload was one group of three requests. The synchronous control was already running; process-proxy tests used a fresh engine start. The table reports direct per-step p50 per request. The one-card async 32K TTFT range uses only the two warm-path repeats.

| workload | sync, one card | async process, one card | async process, two cards (NVLink TP2) |
|---|---|---|---|
| short × 3, step (ms) | 39.6 / 40.1 / 39.4 | 38.3 / 38.1 / 38.3 (−3.5 %) | 37.1 / 37.4 / 36.9 (**−6.5 %**) |
| 32K × 3, step (ms) | 41.2 / 42.2 / 42.3 | 40.2 / 40.6 / 41.3 (−2.4 %) | 39.6 / 39.5 / 40.2 (**−4.5 %**) |
| 32K, 3 concurrent, step (ms) | 75.9 / 76.1 / 76.1 | 75.1 / 75.6 / 75.2 (−0.9 %) | 72.8 / 73.5 / 73.0 (**−3.9 %**) |
| 150K × 2, step (ms) | 41.5 / 42.2 | 40.5 / 39.9 (−3.9 %) | 39.6 / 39.8 (**−5.0 %**) |
| 32K TTFT | 14.56–14.59 s | 14.53–14.54 s | 14.49–14.63 s |
| 150K TTFT | 69.10–69.11 s | 69.24–69.74 s | 68.89–69.09 s |

The one-card async 32K TTFT excludes the first request after a fresh start (15.67 s, lazy module loading; see below).

**Counters over the set:**

- 0 watchdog fires and 0 preempted rounds, against 31 per set for the thread version;
- engine-side submit p50 140 µs, p99 362 µs;
- the first request after a fresh start pays lazy module loading (0.55 s short, about 1 s extra at 32K). A one-request
  warm-up at serve start absorbs the short-prompt cost; the later production check below found that the short warm-up did not cover the long-prefill path.

**Reviewed-code repeat (2026-10-23):** the same TP2 async benchmark set repeated the controlled comparison within noise. It used the same hardware, configuration, contexts, request counts and token budgets; there was no new matched one-card control. Step p50 was:

- short 37.2–37.3 ms;
- 32K 40.4–40.5 ms;
- 3 × 32K 72.7–73.6 ms;
- 150K 39.4–39.9 ms.

Acceptance was at parity. Greedy outputs on 6 fixed prompts were identical between one-card and two-card async. This is a bounded identity check, not a general determinism guarantee.

**Spin and nap are equal** (a spinning variant measured within noise of the napping one at every workload), so the cost
came from running the round inside the engine process, not from the spin. The nap variant frees a core.

### Hardening evidence (2026-10-23)

These are CPU tests with a mock worker and RTX 3090 GPU probes, using synthetic rounds rather than engine contexts. Repetitions and the CPU or GPU boundary are stated in the evidence.

| defect | what it was | fix | evidence (old code → new code) |
|---|---|---|---|
| late-proxy draft corruption | a proxy that lost a round to the watchdog could still write that round's rows into the shared staging the GPU was refilling for the next round: wrong drafts with status OK | headers and resync rows built in private buffers, sent as two parts; the staging is only read | GPU probe: 12 of 12 wrong drafts → 12 of 12 exact, at all three stall points |
| start failure | a proxy that could not start crashed the engine | synchronous rounds with the reason recorded (unless strict start); the shared region is freed on every refusal path | 0 fds / 0 mappings left on all four CUDA refusal paths |
| stall detection | a stopped proxy cost 64 parked rounds | heartbeat every ≤ 100 ms; lost when older than max(2 × deadline, timeout + 250 ms) = 400 ms, or after 3 fires with no round finished | CPU probe: 64 parked rounds (1.40 s) → 3 (0.07 s) at a 20 ms deadline; the slow-but-live control caused no false loss |
| respawn | a lost proxy left the engine target-only for the rest of the run | the GPU stream is drained (event query), the hand-off words reset, and a new proxy starts with a new worker session; at most 3 respawns, restored after an hour | GPU soak 6,000 rounds with a SIGKILL every 2,000: **10,853 drafts exact, 0 wrong**, 3 respawns |
| shutdown | losing a proxy blocked the engine thread (3 s in the test) | non-blocking kill and reap; the speculator closes its round at shutdown | worst submit < 0.3 s |

**Cost of the hardening:** engine-side submit +5.6 to +9.0 µs, about 0.02 % of a 38 ms decode step. Proxy round and
network time are unchanged within run-to-run noise.

**Independent review:**

- It re-ran each probe on the old and new code, plus the paths the test suite did not cover:
  - the process proxy sending over the RDMA transport's client: 0 exact / 12 wrong → 12 / 0;
  - 20,000 real resets against the live C watchdog: no fire;
  - 80 simulated container freezes: 0 false alarms;
  - two more kill soaks: 5,608 and 5,307 drafts exact, 0 wrong.
- The hardening review recorded 222 passed, 1 skipped (196 / 1 before); the later production check recorded 223 passed, 1 skipped.
- The final independent review also cleared the revised GPU ownership and maintenance-window checks. That clearance does not mean the two-card worker has been promoted.

### Supervised fault drills on GB10 (2026-10-23)

On 2026-10-23, the drills used two DGX Spark GB10 nodes, capacity mode with remote Engram and expanded KV, the x86 NVLinked RTX 3090 pair and the process proxy. Each drill ran once. The kill drill used three concurrent 8K-context requests with 3,000-token budgets; the freeze drill used two concurrent requests, with context and output budget not recorded in the comparison summary. The idle check used one request after the pause.

| drill | what happened | result |
|---|---|---|
| `kill -9` of the proxy, 3 concurrent requests | loss detected about 1 s after the signal (process exit); respawn 1 started about 2 s after the signal | 3 / 3 requests completed; the drafter worker kept its process and took a new session |
| `SIGSTOP` of the new proxy, 2 concurrent requests | lost by heartbeat: 531 ms old against the 400 ms limit; the engine killed the stopped process; respawn 2 started about 2 s after the signal | 2 / 2 completed; no stopped or zombie processes left; the engine's shared-memory descriptors unchanged (2) |
| 10 minutes idle, then a request | no false loss (the heartbeat rule only judges while the engine thread runs) | pass |

After both drills:

- the stats showed generation 2 with 1 respawn left;
- there were 3 watchdog fires and 66 target-only rounds, all during the outages;
- the proxy's CPU affinity covered every core.

Switching the drafter from the two-card to the one-card worker took 12.2 s, with the serve decoding target-only in between.

### Production checks (2026-10-23, one-card worker)

Conditions: two DGX Spark GB10 nodes, capacity mode with remote Engram and expanded KV, x86 one-card RTX 3090 worker, process proxy with nap and default lazy loading. These are post-switch checks, not a new matched benchmark. Short requests have two repeats, the first 32K check has one; the greedy check uses fixed prompts with temperature zero and thinking off.

- **Start:** serve ready in 363 s, the same as before (no EAGER).
- **Short requests × 2:** TTFT 0.37 / 0.38 s with no first-request penalty; step p50 38.4 / 38.5 ms.
- **First 32K request after the restart:** TTFT 16.24 s against about 14.5 s warm. That is a one-time lazy-loading cost
  the short warm-up request does not cover. Step p50 was 40.5 ms.
- **Greedy outputs:** 6 / 6 fixed prompts identical to the pre-switch reference.
- **Rank-0 stats at 2,048 rounds:** 2,048 ok, 0 failed, 0 watchdog fires, 0 respawns (3 left), no warning counters.

The warm-up request at start read its 5,041-token prompt in 4.156 s, but returned 1 completion token instead of the
planned 32. The cause is not yet confirmed, and the warm-up may need a longer prompt to cover the 32K loading cost.

The expected effect of this configuration, from the controlled comparison, is −0.9 to −3.9 % step p50 against the historical synchronous control, with warm-path TTFT near parity.

## Upstream claims

No upstream performance claim is used as a measurement in this entry. HeNryous supplies the DGX Spark image and patch lineage; DeepSeek supplies the model and DSpark design; the target uses vLLM. The deadlock reproductions and comparisons above are our tests on that stack.

## Deploy

Install the remote drafter and its matching Spark overlay first. From the repository root, build the helper for the architecture of the engine host:

```sh
cc -O2 -Wall -shared -fPIC \
  -o /data/ofl/libdsv41_hostflag.so \
  code/dspark/spark/remote_dspark/hostflag.c -lpthread
sha256sum /data/ofl/libdsv41_hostflag.so
```

Record both the source and binary digests in your deployment manifest. Mount the helper and transport read-only, and set these variables in the persistent engine environment on both Spark ranks before restarting your engine through its normal launcher:

```sh
export VLLM_DSPARK_REMOTE_ASYNC=1
export VLLM_DSPARK_REMOTE_ASYNC_PROXY=process
export VLLM_DSPARK_REMOTE_ASYNC_SPIN_US=50
export VLLM_DSPARK_REMOTE_ASYNC_NAP_US=20
export VLLM_DSPARK_REMOTE_ASYNC_RESPAWNS=3
export VLLM_DSPARK_REMOTE_ASYNC_RESPAWN_RESET_S=3600
export VLLM_DSPARK_REMOTE_HOSTFLAG_LIB=/data/ofl/libdsv41_hostflag.so
unset CUDA_MODULE_LOADING
```

Keep the remote endpoint, identities, capacity policy and cost calibration from the [one-card worker](https://github.com/ProprietaryLegal/dsv41-remote-dspark-drafter-3090). Confirm `remote DSpark async mode on`, the successful handoff self-test, `mlock` status and `remote DSpark proxy process running`. Check fresh stats for watchdog fires, proxy losses, respawns and worker reconnects. Warm both short and long-prefill paths; the observed short warm-up did not cover the first long request.

For GPU maintenance, keep an exclusive hold throughout a window. Revoke new work, wait for active work to drain, stop the test worker, and check both cards for compute processes while the hold is still held. Refuse to release on an unknown process or failed check. A production worker should hold its locks for its lifetime, and maintenance jobs must use nonblocking acquisition. The generic lock wrapper is in the worker sibling; the full window scripts are not included here.

## Rollback

Remove the async switch from the persistent environment on both ranks, then restart the engine through the same launcher:

```sh
unset VLLM_DSPARK_REMOTE_ASYNC
```

This selects synchronous remote rounds. Keep the endpoint, capacity policy, KV allocation and remote worker in place. Merely changing a shell variable does not reconfigure an already running engine. For a full return to the local drafter, follow the worker sibling's rollback, including restoration of the smaller KV pool.

## Limits and known issues

The included `stage_overlay.sh` checks hashes but assumes the original repository layout and pinned build locations. Configure those paths and pins for your installation before using it.

The first long-prefill request after restart can still pay lazy-loading cost; its cause and the incomplete warm-up are recorded above. Respawn has a bounded budget; once exhausted, capacity mode continues target-only. A refused async startup falls back to sync with a reason unless strict start is enabled. A stuck startup stream must fail loudly rather than free memory still used by the GPU.

The test results qualify this image and build. They do not establish universal stream-memory ordering or deterministic sampling. Supervised recovery drills exercised the two-card worker; production checks exercised one card. No model weights, saved captures or prompts are included.

## Repository layout

| Path | Purpose |
|---|---|
| [code/dspark/spark/remote_dspark](code/dspark/spark/remote_dspark) | Spark client and speculator; `hostflag.c`, `hostflag.py`, `streamsync.py`, `shmregion.py`, `proxy_proc.py`, `roundcore.py` and `round.py` implement handoff, watchdog, process and recovery. |
| [code/dspark/spark/overlay](code/dspark/spark/overlay) | vLLM dispatch overlay, guarded by the remote endpoint switch. |
| [code/dspark/spark/ORDERING.md](code/dspark/spark/ORDERING.md) · [OVERLAP.md](code/dspark/spark/OVERLAP.md) | Ordering argument, races, reset protocol and overlap boundaries. |
| [code/dspark/spark/tests](code/dspark/spark/tests) · [tests_gpu](code/dspark/spark/tests_gpu) | CPU regressions and optional GPU probes. |
| [code/dspark/spark/tools](code/dspark/spark/tools) | Hostflag wait microbenchmark and remote-client diagnostics. |
| [code/results/C/lazy-repro](code/results/C/lazy-repro) | Minimal lazy-loading deadlock reproduction and result note. |
| [code/dspark/CONTRACT.md](code/dspark/CONTRACT.md) | Wire and failure contract needed by the CPU suite. |

## Tests

From a standalone checkout with Python, pytest, NumPy, torch and Triton installed:

```sh
cd code/dspark/spark
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -p no:cacheprovider -q tests
```

The CPU tests use stubs for vLLM and the Triton interpreter. A C compiler is needed for native hostflag tests. The optional ofl integration test needs a built transport; its absence is a skip, not an RDMA qualification.

Publication checks on 2026-10-23 recorded 167 passed / 2 skipped for the earlier Spark copy, then 221 passed / 2 skipped for the updated standalone copy. The later production-tree check recorded 223 passed / 1 skipped, plus 7/7 overlay-staging checks; those counts belong to different trees and dependency setups. Hardware probes under `tests_gpu` require a separate supervised test window and are not part of this CPU command.

## Credits and licences

Our code is Apache-2.0. The included vLLM dispatch overlay retains its Apache-2.0 SPDX header and contributor notice. HeNryous and sudoingX provide the DGX Spark image and patch lineage; DeepSeek provides the model and DSpark drafter. The EXL3 worker uses turboderp/exllamav3 in its sibling package; this entry does not vendor its kernels. See [NOTICE](NOTICE) for component attribution and bundled licence text.

## Related

- dsv41-remote-dspark-drafter-3090: [GitHub](https://github.com/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090).
- dsv41-dspark-drafter-tp2-nvlink: [GitHub](https://github.com/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink).
- ofl-rdma-transport: [GitHub](https://github.com/ProprietaryLegal/ofl-rdma-transport) · [Hugging Face](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport).
- dsv41-longctx-bench-tools: [GitHub](https://github.com/ProprietaryLegal/dsv41-longctx-bench-tools) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools).
- dsv41-spark-offload-findings: [GitHub](https://github.com/ProprietaryLegal/dsv41-spark-offload-findings) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings).
- DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload: [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · [Hugging Face](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).
- dsv41-remote-engram-rdma: [GitHub](https://github.com/ProprietaryLegal/dsv41-remote-engram-rdma) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma).
- dsv41-native-1017-drafter-sm86: [GitHub](https://github.com/ProprietaryLegal/dsv41-native-1017-drafter-sm86) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86).
- dsv41-exact-acceptance-ceiling: [GitHub](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling).
- Image forks: [HeNryous lineage](https://github.com/ProprietaryLegal/deepseek-v41-1017-dgx-spark) · [sudoingX lineage](https://github.com/ProprietaryLegal/dgx-spark-deepseek-v41).
- [Research hub](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload) · [Hugging Face collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e).

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
