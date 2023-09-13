import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp
from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from datasets import load_metric

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint




save_dir = '2023_09_13_focal_loss'

class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if os.path.isfile(os.path.join(self.mask_dir, f))])

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        unique_values = np.unique(mask)
        image = self.transform(image)

        mask_np = np.array(mask)
        masks = [(mask_np == val).astype(int) for val in range(3)]
        mask = np.stack(masks, axis=0)
        mask = torch.tensor(mask, dtype=torch.float32)
        return image, mask


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class LitEfficientNet(pl.LightningModule):
    def __init__(self, train_dataloader=None, val_dataloader=None, test_dataloader=None, batch_size=32):
        super(LitEfficientNet, self).__init__()
        self.train_data_loader = train_dataloader
        self.val_data_loader = val_dataloader
        self.test_data_loader = test_dataloader
        self.model = smp.Unet(
            encoder_name="efficientnet-b7", 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=3
        )
        self.criterion = FocalLoss(alpha=1, gamma=2, ignore_index=0)
        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")
        self.batch_size = batch_size

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.argmax(dim=1)
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.argmax(dim=1)
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.argmax(dim=1)
        preds = self(x)
        loss = F.cross_entropy(preds, y)
        return {'test_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=1e-08, eps=1e-08)

    def train_dataloader(self):
        return self.train_data_loader
    
    def val_dataloader(self):
        return self.val_data_loader
    
    def test_dataloader(self):
        return self.test_data_loader


batch_size = 2
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])
train_data = SkinDataset(root_dir='../../../../Datasets/NewSkinDataset/original/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data = SkinDataset(root_dir='../../../../Datasets/NewSkinDataset/original/val', transform=transform)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
early_stop_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.00, 
    patience=3, 
    verbose=False, 
    mode="min",
)

model = LitEfficientNet(train_dataloader=train_loader ,
                        val_dataloader=test_loader,
                        test_dataloader=test_loader,
                        )

pl.seed_everything(42)
trainer = pl.Trainer(max_epochs=500,
                     callbacks=[early_stop_callback, checkpoint_callback],
                     gpus=[1])
trainer.fit(model)
trainer.save_checkpoint(f'./model_pt/{save_dir}.ckpt')


metric = load_metric("mean_iou")
results = []

train_data = SkinDataset(root_dir='../../../../Datasets/NewSkinDataset/original/train', transform=transform)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

print("inference")
for img_idx, batch in tqdm(enumerate(train_loader)):
    print(img_idx)
    x, y = batch
    x, y = x.to('cpu'), y.to('cpu')
    
    logits = model(x)
    
    probs = torch.nn.functional.softmax(logits, dim=1)
    y_pred_plot = probs.argmax(dim=1).detach().cpu().numpy()
    to_plot = x.detach().cpu().numpy()
    y_true_plot = y.detach().cpu().numpy()

    fig, axs = plt.subplots(3, 3, figsize=(22, 22))
    for i, label in enumerate(['True_Backgroud', 'True Pore', 'True Wrinkle']):
        mask = y_true_plot[0][i]  # 각 클래스에 대한 마스크를 가져옵니다.
        axs[0, i].imshow(mask, cmap='gray')
        axs[0, i].set_title(label)
    
    for i, label in enumerate(['Predicted Background', 'Predicted Pore', 'Predicted Wrinkle']):
        mask = (y_pred_plot[0] == i)  # 예측된 클래스에 대한 마스크 생성
        axs[1, i].imshow(mask, cmap='gray')
        axs[1, i].set_title(label)
    
    for i, label in enumerate(['Background Overlay', 'Pore Overlay', 'Wrinkle Overlay']):
        overlay_image = to_plot[0].transpose((1, 2, 0)).copy()
        mask = (y_pred_plot[0] == i)
        overlay_image[mask] = [255, 0, 0]  # 예측된 영역을 빨간색으로 표시합니다.
        axs[2, i].imshow(overlay_image, cmap='gray')
        axs[2, i].set_title(label)
    
    for ax in axs.ravel():
        ax.axis('off')
    
    plt.savefig(f'result/{save_dir}/image_{img_idx}.png')
    
    # 메트릭 계산 부분
    label_0 = np.where(y_true_plot[0, 0, :, :] == 1, 0, -1)
    label_1 = np.where(y_true_plot[0, 1, :, :] == 1, 1, -1)
    label_2 = np.where(y_true_plot[0, 2, :, :] == 1, 2, -1)

    y_true_combined = np.maximum(label_0, np.maximum(label_1, label_2))

    miou_values = {}
    miou = metric.compute(predictions=[y_pred_plot[0]], 
                          references=[y_true_combined], 
                          num_labels=3, ignore_index=255)['per_category_iou']

    miou_values['img_idx'] = img_idx
    miou_values['background_miou'] = miou[0]
    miou_values['pore_miou'] = miou[1]
    miou_values['wrinkle_miou'] = miou[2]
    
    results.append(miou_values)
    
miou_df = pd.DataFrame(results)
miou_df.to_csv(f'result/{save_dir}/miou.csv')
