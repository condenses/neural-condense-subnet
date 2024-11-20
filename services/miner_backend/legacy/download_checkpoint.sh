dest_dir=checkpoints/
mkdir -p $dest_dir
wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_ft_icae.safetensors?download=true" -P $dest_dir --content-disposition
wget "https://huggingface.co/sggetao/icae/resolve/main/mistral_7b_pretrained_icae.safetensors?download=true" -P $dest_dir --content-disposition