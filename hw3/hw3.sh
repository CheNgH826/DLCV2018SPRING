#wget https://www.dropbox.com/s/8zo086m8smq8zw4/model.h5?dl=1 -O model.h5
python3 infer_test.py model.h5 $1 $2
