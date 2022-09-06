#!/usr/bin/env bash
DOCKER_IMAGE=monai-nvflare:latest

GPU=0

DATA_DIR="${PWD}/data"
INFER_CODE_DIR="${PWD}"  # Location of mammo_inference.py & challenge_evaluate.py
TRAIN_RESULT_DIR="${PWD}/result_server"  # participants result folder
OUTPUT_DIR="${PWD}/test_prediction"  # where to save predictions

# interactive session
#COMMAND="/bin/bash"

# test learner
# dataset_root & datalist_prefix need to be updated for inference on unseen data.
# gt1-3 need updating with the right ground truth.

COMMAND="echo INFERENCE:; \
  python3 /infer_code/mammo_inference.py \
    --model_filepath /result_server/run_1/app_server/best_FL_global_model.pt \
    --dataset_root /data/preprocessed \
    --datalist_prefix /data/dataset_blinded_ \
    --output_root /output; \
  echo EVALUATION:; \
  python3 /infer_code/challenge_evaluate.py \
    --pred /output/test_predictions.json --test_name test_site \
    --site_names test_site \
    --gt1 ../../../dmist_files/dataset_test_site.json"

echo "Starting $DOCKER_IMAGE with GPU=${GPU}"
docker run -it \
--gpus="device=${GPU}" --network=host --ipc=host --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
--name="mammo_inference" \
-e NVIDIA_VISIBLE_DEVICES=${GPU} \
-v ${DATA_DIR}:/data:ro \
-v ${INFER_CODE_DIR}:/infer_code:ro \
-v ${TRAIN_RESULT_DIR}:/result_server:ro \
-v ${OUTPUT_DIR}:/output \
-w /code \
${DOCKER_IMAGE} /bin/bash -c "${COMMAND}"
