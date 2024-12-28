import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTModel, ViTConfig, GPT2Model, GPT2Config, ViTForImageClassification, GPT2LMHeadModel

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

# 生成图形函数
def generate(model, image, text, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        text = text.to(device)
        output = model(image, text)
        generated_text = output.argmax(dim=1)
    return generated_text

# 主函数
def main():
    # 加载模型
    model = CalligraphyModel(num_classes=1000).to('cuda')
    model.load_state_dict(torch.load('calligraphy_model.pth'))

    # 读取新的图片
    image_path = 'new_image.jpg'
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)

    # 输入文字
    text = "这是一段新的文字"

    # 生成图形
    generated_text = generate(model, image, text, 'cuda')
    print("生成的文字:", generated_text)

# 实现可视化
from PIL import ImageDraw, ImageFont

def visualize(image, generated_text):
    # 将生成的文字绘制在图片上
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 36)
    draw.text((10, 10), generated_text, fill=(255, 0, 0), font=font)
    image.show()  # 显示图片
    image.save('generated_image.jpg')  # 保存图片

# 在 generate 函数中调用 visualize 函数
def generate(model, image, text, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        text = text.to(device)
        output = model(image, text)
        generated_text = output.arg

if __name__ == '__main__':
    main()
