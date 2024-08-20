export HF_ENDPOINT="https://hf-mirror.com"
MAX_NUM=27
OUTPUT="result"
INPUT="result"
for ((folder_number = 0; folder_number <= $MAX_NUM; folder_number+=$GPU_COUNT)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id+=1)); do
        current_folder_number=$((folder_number + gpu_id))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        CUDA_VISIBLE_DEVICES=$gpu_id
        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python tools/run.py --seed $SEED --adain --input_root $INPUT --rank $current_folder_number --weight $WEIGHT --output $OUTPUT"
        eval $COMMAND &
        sleep 10
    done
    wait
done