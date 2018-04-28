wget https://github.com/CheNgH826/hello-world/releases/download/dlcv_hw3_8s/8s2_model-03-0.4442-0.873.h5 -o model_best.h5
python3 infer_test.py model_best.h5 $1 $2
