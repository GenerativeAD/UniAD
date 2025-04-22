PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python tools/create_data.py nuscenes --root-path ./data/nuscenes \
#        --out-dir ./data/infos \
#        --extra-tag nuscenes \
#        --version v1.0 \
#        --canbus ./data/nuscenes \
#        --available-scene-names "['scene-0558', 'scene-0098', 'scene-0018', 'scene-1065', 'scene-0906', 'scene-0014', 'scene-0271', 'scene-0553', 'scene-0100', 'scene-0968', 'scene-0270', 'scene-0278', 'scene-0802', 'scene-0103']"

python tools/create_data.py nuscenes --root-path ./data/nuscenes \
       --out-dir ./data/infos_nuscenes_12HZ \
       --extra-tag nuscenes \
       --version interp_12Hz_trainval \
       --canbus ./data/nuscenes \
       --available-scene-names "['scene-0558', 'scene-0098', 'scene-0018', 'scene-1065', 'scene-0906', 'scene-0014', 'scene-0271', 'scene-0553', 'scene-0100', 'scene-0968', 'scene-0270', 'scene-0278', 'scene-0802', 'scene-0103']"