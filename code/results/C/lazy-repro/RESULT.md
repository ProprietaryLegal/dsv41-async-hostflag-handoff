# Async self-test hang (GB10 rank 0, 2026-10-22) — cause and reproduction

- Symptom: `C-capacity-async-cal1` start; rank 0 stopped after `remote DSpark: depth-policy costs …`, main thread at 100 % CPU
  for 25 min, no weight loading; gdb: `at::cuda::sleep` → `cudaLaunchKernel` → `cuLaunchKernel` spinning inside libcuda.
- Cause: `CudaStreamSync.self_test` enqueues `cuStreamWaitValue32` on the main stream, then launches `torch.cuda._sleep` for
  the first time. Under CUDA's default lazy module loading the first launch loads the kernel and waits for the device; the
  device cannot go idle while the main stream is parked on the flag → deadlock.
- Reproduction on x86host GPU-DRAFTER0 (RTX 3090, O2 venv torch), `repro.py`: original self-test under LAZY killed at 60 s;
  original under `CUDA_MODULE_LOADING=EAGER` passes in 0.02 s; fixed self-test (kernels pre-launched before the wait)
  passes in 0.02 s under LAZY and EAGER.
- Runtime exposure: the same hazard applies to any never-launched kernel on the engine thread while a round is parked.
  The async configs therefore set `CUDA_MODULE_LOADING=EAGER`. Airtight alternative for the follow-on queue: release the
  engine stream from pinned host-mapped memory (proxy thread does no CUDA calls), as the Engram spin reader does.

## Second hang (same day, after the self-test fix, with CUDA_MODULE_LOADING=EAGER)
- Startup reached kernel warm-up (`Skipping FlashInfer autotune`, 23:08:44Z) and both ranks stalled > 7 min (sync start: graph
  capture done 6 s later). gdb on rank 0 (`rank0-threads-hang2.txt`): the engine thread spins in `cuModuleLoadData` called
  from Triton's `loadBinary` (a first-use JIT kernel; EAGER does not cover explicit module loads), while another worker
  thread is blocked in `cuMemcpyHtoDAsync` — the proxy thread's reply copy that precedes the flag write. The parked engine
  stream can only be released by that copy + write, and the copy waits behind the module load: deadlock.
- Conclusion: async mode with a CUDA-calling proxy thread is unsafe on GB10 whenever any module loads while a round is in
  flight. Required redesign: the proxy thread makes NO CUDA calls — gather-done and release flags in pinned host-mapped
  memory (engine stream writes / waits on them with cuStreamWriteValue32 / cuStreamWaitValue32), reply read by an engine-
  stream copy enqueued at submit time after the wait. Async configs are not to be run until that lands.
