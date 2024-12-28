import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTConfig, GPT2Model, GPT2Config, ViTForImageClassification, GPT2LMHeadModel, AdamW

# 定义数据集类
class CalligraphyDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        text = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, text

# 定义模型类
class CalligraphyModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CalligraphyModel, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.fc = torch.nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, image, text):
        image_features = self.vit(image).logits
        text_features = self.gpt2(text).last_hidden_state
        combined_features = torch.cat((image_features, text_features), dim=1)
        output = self.fc(combined_features)
        return output

# 训练函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for images, texts in train_loader:
        images, texts = images.to(device), texts.to(device)
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, texts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 主函数
def main():
    # 数据集路径
    csv_file = 'data.csv'
    img_dir = 'images'
    num_classes = 1000  # 根据实际情况调整

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据集和数据加载器
    dataset = CalligraphyDataset(csv_file, img_dir, transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 初始化模型、优化器和损失函数
    model = CalligraphyModel(num_classes).to('cuda')
    optimizer = AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练模型
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, 'cuda')
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'calligraphy_model.pth')

if __name__ == '__main__':
    main()
