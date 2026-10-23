"""Lazy-loading deadlock repro for streamsync.self_test: argv[1] = path to streamsync.py, argv[2] = 'fixed'|'orig'."""
import importlib.util, sys, time, torch
spec = importlib.util.spec_from_file_location("streamsync", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = torch.device("cuda:0")
torch.cuda.set_device(dev)
s = m.CudaStreamSync(dev)
t0 = time.monotonic()
print(sys.argv[2], "self_test ->", s.self_test(), f"{time.monotonic()-t0:.2f}s", flush=True)
