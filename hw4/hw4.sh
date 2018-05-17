wget https://github.com/CheNgH826/hello-world/releases/download/dlcv_hw4/vae_state_model.pth -O VAE/vae_state_model.pth
python3 VAE/vae_infer.py $1 $2
python3 GAN/gan_infer.py $2
python3 ACGAN/acgan_infer.py $2
