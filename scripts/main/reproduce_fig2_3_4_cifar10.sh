# # SAM
python train.py configs/sam.yaml --optimizer.opt_name=sam --model.model_name=wideresnet28_10 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-fig234

python train.py configs/sam.yaml --optimizer.opt_name=sam --model.model_name=wideresnet28_10 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-fig234

# SGD-GrA
python train.py configs/sam.yaml --optimizer.opt_name=samwo --optimizer.group=A --model.model_name=wideresnet28_10 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-fig234

python train.py configs/sam.yaml --optimizer.opt_name=samwo --optimizer.group=A --model.model_name=wideresnet28_10 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-fig234

# SGD-GrB
python train.py configs/sam.yaml --optimizer.opt_name=samwo --optimizer.group=B --model.model_name=wideresnet28_10 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-fig234

python train.py configs/sam.yaml --optimizer.opt_name=samwo --optimizer.group=B --model.model_name=wideresnet28_10 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-fig234


# SGD
python train.py configs/sgd.yaml --optimizer.opt_name=sgd --model.model_name=wideresnet28_10 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.25 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-fig234

python train.py configs/sgd.yaml --optimizer.opt_name=sgd --model.model_name=wideresnet28_10 --model.widen_factor=1 --dataloader.data_name=cifar10 --dataloader.noise=0.5 --dataloader.noise_type=symmetric --trainer.seed=42 --logging.framework_name=wandb --logging.project_name=cifar10-label-noise-fig234