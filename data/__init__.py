import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.retrieval_dataset import dataset_retrieval_eval, dataset_retrieval_train
from data.caption_dataset import dataset_caption_train, dataset_caption_eval
from data.benchmark_dataset import benchmark_caption_eval, benchmark_retrieval_eval
from data.flip_dataset import flip_pretrain
from data.randaugment import RandomAugment
 

def create_dataset(dataset, config, min_scale=0.5):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), 
                                     (0.5, 0.5, 0.5))

    transform_train = transforms.Compose([                        
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 5, isPIL=True, 
                augs=['Identity', 'Brightness','Sharpness','Equalize',
                    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),   
            transforms.ToTensor(),
            normalize,
        ])        

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])  
        
    if dataset=='pretrain':
        dataset = flip_pretrain(config['ann_root'], 
                                   config['image_root'],
                                   transform_train)   
        return dataset  
    
    # caption
    elif dataset=='caption_celeba' or dataset=='caption_celeba_text' or dataset=='caption_face2text':
        train_dataset = dataset_caption_train(transform_train, 
                                            config['image_root'], 
                                            config['ann_root'], 
                                            prompt=config['prompt'])

        val_dataset = dataset_caption_eval(transform_test, 
                                            config['image_root'], 
                                            config['ann_root'], 'val')
        

        test_dataset = dataset_caption_eval(transform_test, 
                                            config['image_root'], 
                                            config['ann_root'], 'test')   
        
        return train_dataset, val_dataset, test_dataset

    elif dataset=='caption_benchmark':   
        test_dataset = benchmark_caption_eval(transform_test, 
                                            config['image_root'], 
                                            config['ann_root'])   
        
        return test_dataset



    # Retrieval 
    elif dataset=='retrieval_celeba' or dataset=='retrieval_celeba_dialog' or dataset=='retrieval_face2text':  
        train_dataset = dataset_retrieval_train(transform = transform_train, 
                                            image_root = config['image_root'], 
                                            ann_root = config['ann_root'], 
                                            dataset = dataset)

        val_dataset = dataset_retrieval_eval(transform = transform_test, 
                                            image_root = config['image_root'], 
                                            ann_root = config['ann_root'], 
                                            dataset = dataset, 
                                            split = 'val')

        test_dataset = dataset_retrieval_eval(transform = transform_test, 
                                            image_root = config['image_root'], 
                                            ann_root = config['ann_root'], 
                                            dataset = dataset,  
                                            split = 'test')   
        
        return train_dataset, val_dataset, test_dataset
    

    elif dataset=='retrieval_benchmark':   
        test_dataset = benchmark_retrieval_eval(transform_test, 
                                            config['image_root'], 
                                            config['ann_root'])   
        return test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, 
                                                      num_replicas=num_tasks, 
                                                      rank=global_rank, 
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,
                                                               samplers, batch_size,
                                                               num_workers,
                                                               is_trains,
                                                               collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False

        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    


if __name__ == "__main__":
    import argparse 
    import ruamel.yaml as yaml
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/retrieval_celeba.yaml')
    args = parser.parse_args()

    yml = yaml.YAML(typ='rt')
    config = yml.load(open(args.config, 'r'))
    create_dataset("retrieval_celeba", config, min_scale=0.5)