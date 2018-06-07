wget https://www.dropbox.com/s/ezm323d4mss2lhz/32s_model.h5?dl=1 -O model_best.h5
python3 infer_test.py model_best.h5 $1 $2
