import torch
import torchvision
from torchvision import transforms

# 图像变换
transform = transforms.Compose([
    transforms.Resize((50, 50)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
data = torchvision.datasets.ImageFolder(root='/mnt/data/jiachen/small_fov_seg/train/sam_cls_data', transform=transform)
loader = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=True)

# 加载预训练模型
model = torchvision.models.resnet50(pretrained=True)
num_classes = len(data.classes)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)

# 模型训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 10

for epoch in range(num_epochs):
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

torch.save(model, "resnet50_smallfov_cls.pth")
# 模型测试
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: {:.2f}%'.format(correct / total * 100))