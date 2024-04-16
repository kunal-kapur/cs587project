import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from books_dataloader import BooksDataset
from tqdm import tqdm
import torch
from models import DCN
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import os

NUM_NUMERICAL_FEATURES = 2

LR = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 5
DEEP_LAYERS = 1


# for no MLP

"""
For no cross layers currently doing 
[600, 800, 600]

With single cross layer doing
[600,300] over half a reduction in weights

"""

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

books_data = BooksDataset(path="data/books_dataset.csv")

train_dataset, val_dataset= torch.utils.data.random_split(books_data, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset,
                         batch_size=BATCH_SIZE, shuffle=True)

val_dataloader = DataLoader(val_dataset,
                         batch_size=BATCH_SIZE, shuffle=True)

category_list = books_data.get_category_list()

model = DCN(categorical_features=category_list, num_numerical_features=NUM_NUMERICAL_FEATURES,
            dcn_layer_len=DEEP_LAYERS, layer_sizes=[50, 100, 50], concat_layer_sizes=[25], output_dim=4).to(device=device)

#loss_fn = nn.MSELoss(reduction='sum')
loss_fn = nn.NLLLoss()
#loss_fn = nn.CrossEntropyLoss()

opt = Adam(params=model.parameters(), lr=LR)

epoch_list, train_loss_list, val_loss_list = [], [], []
training_accuracy_list, validation_accuracy_list = [], []
for epoch in tqdm(range(NUM_EPOCHS)):
    print("EPOCH:", epoch)
    epoch_list.append(epoch)
    total_train_loss = 0
    total_val_loss = 0
    num_training_correct, num_validation_correct, total = 0, 0, 0
    model.train() # set model to training mode
    for cat_values, numerical_values, rating in tqdm(train_dataloader):
        cat_values = cat_values.to(device)
        numerical_values = numerical_values.to(device)
        rating = rating.to(device)

        pred = model.forward(categorical_input=cat_values, numerical_input=numerical_values)
        loss = loss_fn(pred, rating)
        opt.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
        opt.step()

        total_train_loss += loss
        num_training_correct += torch.sum(torch.argmax(pred, dim=1) == rating)
        total += (cat_values.shape[0])

    train_loss_list.append((total_train_loss).item())
    training_accuracy_list.append((num_training_correct / total).item())
    total = 0
    with torch.no_grad():
        model.eval() # set model to evaluation mode
        for cat_values, numerical_values, rating in tqdm(val_dataloader):
            cat_values = cat_values.to(device)
            numerical_values = numerical_values.to(device)
            rating = rating.to(device)
            pred = model.forward(categorical_input=cat_values, numerical_input=numerical_values)

            loss = loss_fn(pred, rating)

            total_val_loss += loss
            num_validation_correct += torch.sum(torch.argmax(pred, dim=1) == rating)
            total += (cat_values.shape[0])

    val_loss_list.append((total_val_loss).item())
    validation_accuracy_list.append((num_validation_correct / total).item())

    print("train loss", train_loss_list[-1])
    print("Training accuracy", training_accuracy_list[-1])
    print("validation loss", val_loss_list[-1])
    print("Validation accuracy", validation_accuracy_list[-1])
    #print(f"{num_correct}/{total}", num_correct/total)


fig1, ax1 = plt.subplots()
ax1.plot(epoch_list,train_loss_list, label='train loss')
ax1.plot(epoch_list, val_loss_list, label='validation loss')
# Add a title and labels
ax1.set_title('Training curve')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss')
# Show the plot
ax1.legend()

# second plot
fig2, ax2 = plt.subplots()
ax2.plot(epoch_list,training_accuracy_list, label='train accuracy')
ax2.plot(epoch_list, validation_accuracy_list, label='validation accuracy')
# Add a title and labels
ax2.set_title('Training curve')
ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
# Show the plot
ax2.legend()


path = f'curve_plot_{DEEP_LAYERS}DeepLayers_{NUM_EPOCHS}epochs'

if not os.path.exists(path=path):
    os.mkdir(path=path)

fig1.savefig(f'{path}/loss_plot.png')
fig2.savefig(f'{path}/accuracy_plot.png')

with open(f"{path}/results.txt", "w") as f:
    f.write("EPOCHS,Training Loss,Validation Loss\n")
    for i in range(len(epoch_list)):
        f.write(f"{epoch_list[i]},{train_loss_list[i]},{val_loss_list[i]}\n")