import torch
import torch.nn as nn
import torch.optim as optim



from torch.utils.data import DataLoader

from src.dataset.MI_dataset_all_subjects import MI_Dataset as MI_Dataset_all_subjects
from src.dataset.MI_dataset_single_subject import MI_Dataset as MI_Dataset_single_subject

from config.default import cfg

from models.eegnet import EEGNetv4
from models.eegnet2 import EEGNet
from models.my import MyModel

from utils.eval import accuracy

device = 'cpu'# torch.device('cuda' if torch.cuda.is_available() else 'cpu')


subject = 1
train_runs = [0,1,2,3,4]
test_runs = [5]


train_dataset = MI_Dataset_single_subject(subject, train_runs, device = device)
test_dataset = MI_Dataset_single_subject(subject, test_runs, device = device)

train_dataloader = DataLoader(train_dataset,  batch_size=cfg['train']['batch_size'], shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_dataset,  batch_size=cfg['train']['batch_size'], shuffle=False, drop_last=True)


# train_subjects = [1,2,3,4,5,6,7,8]
# test_subjects = [9]



# train_dataset = MI_Dataset_all_subjects(train_subjects)
# test_dataset = MI_Dataset_all_subjects(test_subjects)

# train_dataloader = DataLoader(train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
# test_dataloader = DataLoader(test_dataset, batch_size=cfg['train']['batch_size'], shuffle=False)

model = MyModel([64, 22, 401])
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['learning_rate'], weight_decay=cfg['train']['weight_decay'])

# Training loop
for epoch in range(cfg['train']['n_epochs']):
    epoch_loss = 0.0

    for batch_features, batch_labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 10 == 9:
        train_accuracy = accuracy(model, train_dataloader)
        test_accuracy = accuracy(model, test_dataloader)
        print(f"Epoch {epoch + 1}/{cfg['train']['n_epochs']}, Loss: {epoch_loss}, Train accuracy: {train_accuracy:.2f}%, Test accuracy: {test_accuracy:.2f}%")

print("#"*50)
print(f'Final_loss: {epoch_loss}')
print(f'Final train accuracy: {accuracy(model, train_dataloader):.2f}%')
print(f'Final test accuracy: {accuracy(model, test_dataloader):.2f}%')