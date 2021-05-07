

from utils import *
import wandb
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import pandas as pd
import numpy as np
from mapcalc import *
from mapcalc import calculate_map, calculate_map_range
import dataset
import pdb
from dataset import *


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


batchsize = 2
in_dim = (300,300)
normalization_data = torch.load('mean-std.pt')
num_classes = 7
print_every = 10

diseases = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
mapping = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
inv_mapping = {mapping[k]:k for k in mapping.keys()}


def calculate_AP(model, data_loader, metric, idx = 'val_mAP', th=0.5):
    """
    Calculates and stores the average precision in the the metrics dictionary.

    model: (nn.Module) model
    data_loader: (nn.DataLoader) Dataloader
    metric: (Dictionary) Dictionary with Average/Class Meter
    
    Returns mAP over all classes for IOU threshold of 0.5
    """
    model.eval()
    for dis_in, disease in enumerate(diseases):
        for i, data in enumerate(data_loader):
            image, target = data
            image = image.to(device)
            target = {k: v.numpy()[0] for k, v in target.items()}
            class_id = inv_mapping[target['labels'].item()]
            if class_id == disease:
                result = model(image)[0]
                result = {k: v.cpu() for k, v in result.items()}
                mAP = calculate_map(target, result, th)
                metric[idx].update(dis_in, mAP, n=1)
    return  metric[idx].class_average()


def initialize_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model


def train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_valid, data_loader_test, diseases, config):
    best_mAP = 0
    header = 10 * '='
    short_header = 5*'='
    metrics = {'val_mAP': ClassMeter(diseases)}
    metrics['train_loss'] = AverageMeter()
    for epoch in range(config.epochs):
        header = 10 * '='
        short_header = 5*'='
        print(header, "Epoch {}".format(epoch), header)
        for i, d in enumerate(data_loader_train):

            # Header
            header = 10*"="
            short_header = 5 * '='

            # Training
            model.train()

            image, target = d
            image = torch.stack([im.to(device) for im in image])
            target = [{k: v.to(device) for k, v in t.items()} for t in target]
            losses = model(image, target)
            loss = sum(loss for loss in losses.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            metrics['train_loss'].update(loss.item(), n=len(target))

            # Log to weights and biases
            wandb.log({'train_loss': metrics['train_loss'].avg})

            # Print every 10 epochs
            if i % (print_every) == 0:
                print('[Epoch {} | {} {}] Training Loss: {}'.format(epoch, i, len(data_loader_train), metrics['train_loss'].avg))

        with torch.no_grad():
            calculate_AP(model, data_loader_valid, metrics)
            
            # Print Summary
            print(short_header, "Validation", short_header)
            current_mAP = metrics['val_mAP'].class_average()
            print('[Val average mAP] : {}'.format(current_mAP))                                                                                                                           
            # Log Values
            wandb.log({'val_mAP': current_mAP, 'epoch':epoch})
            for idx, disease in enumerate(diseases):
                wandb.log({'val_mAP' + '_' +disease: metrics['val_mAP'].avg[idx], 'epoch': epoch})

            # SAVE BEST MODEL
            if current_mAP > best_mAP:
                torch.save(model.state_dict(), wandb.run.dir +
                        '/best.pt'.format(epoch))
                best_mAP = current_mAP

            # SAVE EVERY FIVE EPOCHS
            if epoch % 5 == 0:
                torch.save(model.state_dict(), wandb.run.dir +
                        '/epoch_{}.pt'.format(epoch))

    # fig = draw_box(image, image_id, bbox, target=None, confidence=None)
    # wandb.log({'sample_figure': fig, 'epoch':epoch})

    # Test mAP
    test_metrics = {'test_mAP': ClassMeter(diseases)}
    with torch.no_grad():
        calculate_AP(model, data_loader_test, test_metrics, idx='test_mAP')
    print(header, "Test", header)
    test_mAP = test_metrics['test_mAP'].class_average()
    print('[Test average mAP] : {}'.format(test_mAP))

    # LOG STUFF
    wandb.log({'test_mAP': test_mAP, 'epoch':epoch})
    for idx, disease in enumerate(diseases):
        wandb.log({'test_mAP' + '_' + disease: test_metrics['test_mAP'].avg[idx], 'epoch':epoch})
        


        


# %%
if __name__ == "__main__":
    hyperparameter_defaults = dict(
    num_workers=0,
    batch_size=2,
    learning_rate=0.001,
    epochs=10,)
    wandb.init(project = 'FSDL', config=hyperparameter_defaults )
    config = wandb.config
    dataset = SkinData('/data/kevinmiao', 'final.csv', transform=transforms.Compose([ToTensor, Normalizer(normalization_data)]))
    train_data, test_data, valid_data = torch.utils.data.random_split(dataset,[int(0.7 * len(dataset)), int(0.15 * len(dataset)), int(0.15 * len(dataset))+1],  generator=torch.Generator().manual_seed(42))
    data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size, collate_fn = collate_fn)
    data_loader_test = torch.utils.data.DataLoader(test_data, batch_size=1)
    data_loader_valid =  torch.utils.data.DataLoader(valid_data, batch_size=1)
    model = initialize_model()
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[15, 30], gamma=0.1, last_epoch=-1)
    train_model(model, optimizer, lr_scheduler, data_loader_train, data_loader_valid, data_loader_test, diseases, config)
            



            




