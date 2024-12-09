download:
	mkdir -p pretrain && cd pretrain && \
	wget https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper.pth && \
	wget https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth && \
	cd .. && \
	mkdir -p finetune && cd finetune && \
	wget https://huggingface.co/unacanal/dacon-236420/resolve/main/net_d_5000.pth && \
	wget https://huggingface.co/unacanal/dacon-236420/resolve/main/net_g_5000.pth


train_ddp:
	CUDA_VISIBLE_DEVICES=0,1,2,3 \
	python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=3721 basicsr/train.py \
		-opt options/train_dacon.yml --auto_resume --launcher pytorch

train:
	python3 basicsr/train.py -opt options/train_dacon_test.yml --auto_resume

test:
	python basicsr/test.py -opt options/test_dacon.yml