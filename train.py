import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from books_dataloader import BooksDataset
from tqdm import tqdm
import torch
from models import DCN
from torch import nn
from torch.optim import Adam

NUM_NUMERICAL_FEATURES = 2
EMBEDDING_DIM  = 100

LR = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

books_data = BooksDataset(path="data/books_dataset.csv")
dataloader = DataLoader(torch.utils.data.Subset(books_data, indices=range(0,10000)),
                         batch_size=BATCH_SIZE, shuffle=True)

category_list = books_data.get_category_list()

model = DCN(categorical_features=category_list, num_numerical_features=NUM_NUMERICAL_FEATURES,
            embedding_dim=EMBEDDING_DIM, 
            dcn_layer_len=3, layer_sizes=[300, 200, 50], output_dim=1).to(device=device)

loss_fn = nn.MSELoss()
opt = Adam(params=model.parameters(), lr=LR)

epoch_list, train_loss_list = [], []
for epoch in tqdm(range(NUM_EPOCHS)):
    print("EPOCH:", epoch)
    epoch_list.append(epoch)
    total_train_loss = 0
    for cat_values, numerical_values, rating in tqdm(dataloader):

        cat_values = cat_values.to(device)
        numerical_values = numerical_values.to(device)
        rating = rating.to(device)

        pred = model.forward(categorical_input=cat_values, numerical_input=numerical_values)

        loss = loss_fn(pred, rating.float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_train_loss += loss
    print("loss", total_train_loss)
    train_loss_list.append(total_train_loss)
print(train_loss_list)