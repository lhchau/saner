from .cifar100 import get_cifar100
from .cifar10 import get_cifar10
from .miniwebvision import get_miniwebvision
from .animal10n import get_animal10n
from .tiny_imagenet import get_tiny_imagenet
from .cifar10_jo import get_cifar10_jo
from .cifar100_jo import get_cifar100_jo
from .cifar100_index import get_cifar100_index
from .cifar10_index import get_cifar10_index


def get_dataloader(
    data_name='cifar10',
    batch_size=256,
    num_workers=4,
    noise=0.25,
    noise_type='symmetric',
    resize_image=224,
    data_augmentation="standard",
    data_size=1,
    use_val=False,
    samples_per_class=None
):
    print('==> Preparing data..')

    if data_name == "cifar100":
        return get_cifar100(batch_size, num_workers, noise, noise_type, data_augmentation, data_size=data_size, use_val=use_val, samples_per_class=samples_per_class)
    elif data_name == "cifar10":
        return get_cifar10(batch_size, num_workers, noise, noise_type, data_augmentation, data_size=data_size, use_val=use_val)
    elif data_name == "cifar10_jo":
        return get_cifar10_jo(batch_size, num_workers, noise, noise_type, data_augmentation, data_size=data_size, use_val=use_val)
    elif data_name == "cifar10_index":
        return get_cifar10_index(batch_size, num_workers, noise, noise_type, data_augmentation, data_size=data_size, use_val=use_val)
    elif data_name == "cifar100_jo":
        return get_cifar100_jo(batch_size, num_workers, noise, noise_type, data_augmentation, data_size=data_size, use_val=use_val)
    elif data_name == "cifar100_index":
        return get_cifar100_index(batch_size, num_workers, noise, noise_type, data_augmentation, data_size=data_size, use_val=use_val)
    elif data_name == "miniwebvision":
        return get_miniwebvision(batch_size, num_workers, resize_image)
    elif data_name == "tiny_imagenet":
        return get_tiny_imagenet(batch_size, num_workers, noise, noise_type)
    elif data_name == "animal10n":
        return get_animal10n(batch_size, num_workers)