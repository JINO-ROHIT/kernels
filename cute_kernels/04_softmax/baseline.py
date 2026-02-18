#ref - https://docs.pytorch.org/docs/main/generated/torch.cuda.Event.html#torch.cuda.Event

import torch

t = torch.randn(1000, 32000, device='cuda', dtype=torch.float32)

# warmup
for _ in range(10):
    _ = torch.nn.functional.softmax(t, dim=-1)

torch.cuda.synchronize()

num_iters = 100

start_event = torch.cuda.Event(enable_timing = True)
end_event = torch.cuda.Event(enable_timing = True)

start_event.record()
for _ in range(num_iters):
    _ = torch.nn.functional.softmax(t, dim=-1)
end_event.record()

torch.cuda.synchronize()

elapsed_time_ms = start_event.elapsed_time(end_event)
elapsed_time_sec = elapsed_time_ms / 1000.0

print(f"softmax took {elapsed_time_sec / num_iters * 1000:.4f} ms per iteration")