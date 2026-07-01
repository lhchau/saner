####
python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar10 \
  --dataloader.noise=0.25 \
  --dataloader.samples_per_class=None \
  --trainer.seed=42 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar10-sam-noise-bootstrap

python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar10 \
  --dataloader.noise=0.25 \
  --dataloader.samples_per_class=None \
  --trainer.seed=43 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar10-sam-noise-bootstrap

python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar10 \
  --dataloader.noise=0.25 \
  --dataloader.samples_per_class=None \
  --trainer.seed=44 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar10-sam-noise-bootstrap


python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar10 \
  --dataloader.noise=0.5 \
  --dataloader.samples_per_class=None \
  --trainer.seed=42 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar10-sam-noise-bootstrap

python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar10 \
  --dataloader.noise=0.5 \
  --dataloader.samples_per_class=None \
  --trainer.seed=43 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar10-sam-noise-bootstrap

python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar10 \
  --dataloader.noise=0.5 \
  --dataloader.samples_per_class=None \
  --trainer.seed=44 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar10-sam-noise-bootstrap
  

####
python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar100 \
  --dataloader.noise=0.25 \
  --dataloader.samples_per_class=None \
  --trainer.seed=42 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar100-sam-noise-bootstrap

python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar100 \
  --dataloader.noise=0.25 \
  --dataloader.samples_per_class=None \
  --trainer.seed=43 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar100-sam-noise-bootstrap

python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar100 \
  --dataloader.noise=0.25 \
  --dataloader.samples_per_class=None \
  --trainer.seed=44 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar100-sam-noise-bootstrap


python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar100 \
  --dataloader.noise=0.5 \
  --dataloader.samples_per_class=None \
  --trainer.seed=42 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar100-sam-noise-bootstrap

python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar100 \
  --dataloader.noise=0.5 \
  --dataloader.samples_per_class=None \
  --trainer.seed=43 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar100-sam-noise-bootstrap

python baselines/train_bootstrap.py configs/sam.yaml \
  --optimizer.opt_name=saner \
  --optimizer.rho=0.1 \
  --optimizer.alpha=0.5 \
  --trainer.alpha_scheduler=50 \
  --model.model_name=resnet18 \
  --model.widen_factor=1 \
  --dataloader.data_name=cifar100 \
  --dataloader.noise=0.5 \
  --dataloader.samples_per_class=None \
  --trainer.seed=44 \
  --logging.framework_name=wandb \
  --logging.project_name=cifar100-sam-noise-bootstrap
  