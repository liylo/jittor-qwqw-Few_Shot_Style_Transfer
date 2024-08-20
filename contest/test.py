import subprocess
import os

seeds=[ s[:-4] for s in os.listdir("latents") ]
taskid=[ t for t in os.listdir("config") ]
weights=[ w for w in os.listdir("weights") ]
weights.remove("oft2")
gpu_count = 8

for s in seeds:
    for w in weights:
        subprocess.run(
            ["bash", "scripts/raw.sh"],
            env={**os.environ, "SEED": str(s), "WEIGHT": str(w), "GPU_COUNT": str(gpu_count)}
        )
for s in seeds:
    for w in weights:
        subprocess.run(
            ["bash", "scripts/adain.sh"],
            env={**os.environ, "SEED": str(s), "WEIGHT": str(w), "GPU_COUNT": str(gpu_count)}
        )
for s in seeds:
    subprocess.run(
        ["bash", "scripts/a.sh"],
        env={**os.environ, "SEED": str(s), "GPU_COUNT": str(gpu_count)}
    )

subprocess.run(
            ["python", "tools/resize.py"],
        )
