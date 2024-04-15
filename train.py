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
EMBEDDING_DIM  = 100

LR = 0.0001
BATCH_SIZE = 64
NUM_EPOCHS = 1
DEEP_LAYERS = 1


# for no MLP

"""
For no cross layers currently doing 
[600, 800, 600]

With single cross layer doing
[600,300] over half a reduction in weights

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

books_data = BooksDataset(path="data/books_dataset.csv")

train_dataset, val_dataset= torch.utils.data.random_split(books_data, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset,
                         batch_size=BATCH_SIZE, shuffle=True)

val_dataloader = DataLoader(val_dataset,
                         batch_size=BATCH_SIZE, shuffle=True)

category_list = books_data.get_category_list()

model = DCN(categorical_features=category_list, num_numerical_features=NUM_NUMERICAL_FEATURES,
            embedding_dim=EMBEDDING_DIM, 
            dcn_layer_len=DEEP_LAYERS, layer_sizes=[600, 300], concat_layer_sizes=[400, 100], output_dim=1, 
            embed_by_category=True).to(device=device)

loss_fn = nn.MSELoss(reduction='sum')
opt = Adam(params=model.parameters(), lr=LR)

epoch_list, train_loss_list, val_loss_list = [], [], []
for epoch in tqdm(range(NUM_EPOCHS)):
    print("EPOCH:", epoch)
    epoch_list.append(epoch)
    total_train_loss = 0
    total_val_loss = 0
    num_correct, total = 0, 0
    model.train()
    for cat_values, numerical_values, rating in tqdm(train_dataloader):
        cat_values = cat_values.to(device)
        numerical_values = numerical_values.to(device)
        rating = rating.to(device)

        pred = model.forward(categorical_input=cat_values, numerical_input=numerical_values)

        loss = loss_fn(pred, rating.float())
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_train_loss += loss
        # num_correct += torch.sum(pred.round() == rating)
        total += (cat_values.shape[0])

    train_loss_list.append((total_train_loss / total).item())
    total = 0
    with torch.no_grad():
        for cat_values, numerical_values, rating in tqdm(val_dataloader):
            cat_values = cat_values.to(device)
            numerical_values = numerical_values.to(device)
            rating = rating.to(device)
            pred = model.forward(categorical_input=cat_values, numerical_input=numerical_values)

            loss = loss_fn(pred, rating.float())

            total_val_loss += loss
            # num_correct += torch.sum(pred.round() == rating)
            total += (cat_values.shape[0])

    val_loss_list.append((total_val_loss / total).item())

    print("train loss", train_loss_list[-1])
    print("validation loss", val_loss_list[-1])
    #print(f"{num_correct}/{total}", num_correct/total)

plt.plot(epoch_list,train_loss_list, label='train loss')
plt.plot(epoch_list, val_loss_list, label='validation loss')


# Add a title and labels
plt.title('Training curve')
plt.xlabel('epoch')
plt.ylabel('loss')

# Show the plot
plt.legend()


path = f'curve_plot_{DEEP_LAYERS}DeepLayers_{NUM_EPOCHS}epochs'

if not os.path.exists(path=path):
    os.mkdir(path=path)
    plt.savefig(f'{path}/plot.png')

with open(f"{path}/results.txt", "w") as f:
    f.write("EPOCHS," + ",".join(map(str, epoch_list)) + "\n")
    f.write("Validation Loss," + ",".join(map(str, val_loss_list)) + "\n")
    f.write("Training Loss," + ",".join(map(str, train_loss_list)) + "\n")