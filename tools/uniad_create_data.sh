PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python tools/create_data.py nuscenes --root-path ./data/nuscenes \
#        --out-dir ./data/infos \
#        --extra-tag nuscenes \
#        --version v1.0 \
#        --canbus ./data/nuscenes \

python tools/create_data.py nuscenes --root-path ./data/nuscenes \
       --out-dir ./data/infos_nuscenes_12HZ \
       --extra-tag nuscenes \
       --version interp_12Hz_trainval \
       --canbus ./data/nuscenes \
       --available-scene-names "['scene-0014']"