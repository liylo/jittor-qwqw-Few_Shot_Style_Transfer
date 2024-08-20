import subprocess
import os
weights=[ w for w in os.listdir("weights") ]
weights.remove("oft2")
gpu_count = 1
subprocess.run(
        ["bash", "scripts/train.sh"],
        env={**os.environ, "OUTPUT_DIR_PREFIX": "train_weights/"+"oft8"+"/", "LEARNING_RATE": "1e-4","LR_SCHEDULER": "constant", "MAX_TRAIN_STEPS": "500","LORA":"oft","R": "8", "GPU_COUNT": str(gpu_count)}
    )
subprocess.run(
        ["bash", "scripts/train.sh"],
        env={**os.environ, "OUTPUT_DIR_PREFIX": "train_weights/"+"oft16"+"/", "LEARNING_RATE": "1e-4","LR_SCHEDULER": "constant", "MAX_TRAIN_STEPS": "500","LORA":"oft","R": "16", "GPU_COUNT": str(gpu_count)}
    )
subprocess.run(
        ["bash", "scripts/train.sh"],
        env={**os.environ, "OUTPUT_DIR_PREFIX": "train_weights/"+"dora4"+"/", "LEARNING_RATE": "1e-4","LR_SCHEDULER": "constant", "MAX_TRAIN_STEPS": "500","LORA":"dora","R": "4", "GPU_COUNT": str(gpu_count)}
    )
subprocess.run(
        ["bash", "scripts/train.sh"],
        env={**os.environ, "OUTPUT_DIR_PREFIX": "train_weights/"+"dora8"+"/", "LEARNING_RATE": "1e-4","LR_SCHEDULER": "constant", "MAX_TRAIN_STEPS": "500","LORA":"dora","R": "8", "GPU_COUNT": str(gpu_count)}
    )
subprocess.run(
        ["bash", "scripts/train.sh"],
        env={**os.environ, "OUTPUT_DIR_PREFIX": "train_weights/"+"oft2"+"/", "LEARNING_RATE": "3e-5","LR_SCHEDULER": "cosine", "MAX_TRAIN_STEPS": "1500","LORA":"oft","R": "2", "GPU_COUNT": str(gpu_count)}
    )