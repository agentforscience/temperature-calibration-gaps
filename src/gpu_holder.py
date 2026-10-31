"""Reserve GPU memory in 1 GB chunks until a RELEASE file appears (shared-GPU host).
Target GB read from logs/holder/TARGET each loop; exits on logs/holder/RELEASE."""
import os, sys, time, torch
gpu = int(sys.argv[1]); d = sys.argv[2]
torch.cuda.set_device(gpu)
chunks = []
while not os.path.exists(f"{d}/RELEASE"):
    try: target = float(open(f"{d}/TARGET").read())
    except Exception: target = 40
    if len(chunks) < target:
        try: chunks.append(torch.empty(1 << 30, dtype=torch.uint8, device="cuda"))
        except Exception: time.sleep(0.5)
    elif len(chunks) > target: chunks.pop(); torch.cuda.empty_cache()
    else: time.sleep(0.2)
    open(f"{d}/HELD", "w").write(str(len(chunks)))
