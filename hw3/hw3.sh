wget https://github.com/CheNgH826/hello-world/releases/download/dlcv_hw3/32s_model-09-0.3737.h5 -o model.h5
python3 infer_test.py model.h5 $1 $2
