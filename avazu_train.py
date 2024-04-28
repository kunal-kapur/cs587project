import torch.utils
from torch.utils.data import DataLoader
import torch.utils.data
from dataloader import AvazuDataSet
from tqdm import tqdm
import torch
from models import DCN
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt
# import seaborn as sns
from absl import app, flags
import os

FLAGS = flags.FLAGS

flags.DEFINE_integer("cross_layers", 0, "Number of cross layers in the model")
flags.DEFINE_integer("batch", 512, "Batch size")
flags.DEFINE_integer("epochs", 15, "Number of epochs to run")
flags.DEFINE_float("lr", 0.00001, "Learning Rate")
flags.DEFINE_list("deep_layers", [500,800,500], "Sizes of the deep (mlp) layers")
flags.DEFINE_bool("stacked", False, "Use a stacked or parallel architecture")
flags.DEFINE_float("reg", 0, "Regularization term")
flags.DEFINE_enum("data", "avazu", ['avazu', 'books'], "Dataset to pick")
flags.DEFINE_bool("v2", False, "If we use DCN version 2 or version 1")


NUM_NUMERICAL_FEATURES = 0

# for no MLP
def main(argv):
    LR = FLAGS.lr
    BATCH_SIZE = FLAGS.batch
    NUM_EPOCHS = FLAGS.epochs
    CROSS_LAYERS = FLAGS.cross_layers
    DEEP_LAYERS = [int(i) for i in FLAGS.deep_layers]
    CONCAT_LAYERS = []
    STACKED = FLAGS.stacked
    LMBD = FLAGS.reg
    OUTPUT_DIM = 2
    if (FLAGS.data == 'avazu'):
        PATH = "data/avazu_dataset.csv"
    else:
        exit()

    torch.manual_seed(15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avazu_data = AvazuDataSet(path="data/avazu_dataset.csv")

    train_dataset, val_dataset= torch.utils.data.random_split(avazu_data, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset,
                            batch_size=BATCH_SIZE, shuffle=True)

    val_dataloader = DataLoader(val_dataset,
                            batch_size=BATCH_SIZE, shuffle=True)

    category_list = avazu_data.get_category_list()

    model = DCN(categorical_features=category_list, num_numerical_features=NUM_NUMERICAL_FEATURES,
                dcn_layer_len=CROSS_LAYERS, layer_sizes=DEEP_LAYERS, concat_layer_sizes=CONCAT_LAYERS, output_dim=OUTPUT_DIM,
                cross_net_V2=FLAGS.v2, stacked=STACKED).to(device=device)
    print("Printing trainable parameters")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, len(param.data))

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
        for cat_values, numerical_values, rating in (train_dataloader):

            cat_values = cat_values.to(device)
            numerical_values = numerical_values.to(device)
            rating = rating.type(torch.int64).to(device)

            pred = model.forward(categorical_input=cat_values, numerical_input=numerical_values)
            loss = loss_fn(pred, rating) + model.get_regularization_term(lmbd=LMBD)
            opt.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            opt.step()

            total_train_loss += loss
            num_training_correct += torch.sum(torch.argmax(pred, dim=1) == rating)
            total += (cat_values.shape[0])

        train_loss_list.append((total_train_loss).item() / total)
        training_accuracy_list.append((num_training_correct / total).item())
        total = 0
        with torch.no_grad():
            model.eval() # set model to evaluation mode
            for cat_values, numerical_values, rating in (val_dataloader):
                cat_values = cat_values.to(device)
                numerical_values = numerical_values.to(device)
                rating = rating.type(torch.int64).to(device)
                pred = model.forward(categorical_input=cat_values, numerical_input=numerical_values)
                loss = loss_fn(pred, rating)
                total_val_loss += loss
                num_validation_correct += torch.sum(torch.argmax(pred, dim=1) == rating)
                total += (cat_values.shape[0])

        val_loss_list.append((total_val_loss).item() / total)
        validation_accuracy_list.append((num_validation_correct / total).item())

        print("train loss", train_loss_list[-1])
        print("Training accuracy", training_accuracy_list[-1])
        print("validation loss", val_loss_list[-1])
        print("Validation accuracy", validation_accuracy_list[-1])

        # if len(train_loss_list) > 1 and train_loss_list[-1] == train_loss_list[-2]:
        #     print("EARLY STOP")
        #     break


    fig1, ax1 = plt.subplots()
    ax1.plot(epoch_list,train_loss_list, label='train loss')
    ax1.plot(epoch_list, val_loss_list, label='validation loss')
    # Add a title and labels
    ax1.set_title(f'Avazu Training curve, cross layers:{CROSS_LAYERS}')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    # Show the plot
    ax1.legend()

    # second plot
    fig2, ax2 = plt.subplots()
    ax2.plot(epoch_list,training_accuracy_list, label='train accuracy')
    ax2.plot(epoch_list, validation_accuracy_list, label='validation accuracy')
    # Add a title and labels
    ax2.set_title(f'Avazu Training curve, cross layers:{CROSS_LAYERS}')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    # Show the plot
    ax2.legend()


    if not os.path.exists("avazu_results"):
        os.mkdir("avazu_results")

    path = f'avazu_results/curve_plot_{CROSS_LAYERS}crossLayers__{str(DEEP_LAYERS)}_deepLayers{str(CONCAT_LAYERS)}concatLayers_{NUM_EPOCHS}epochs{LR}LR{LMBD}lambda{STACKED}stacked{FLAGS.v2}isv2'

    if not os.path.exists(path=path):
        os.mkdir(path=path)

    fig1.savefig(f'{path}/loss_plot.png')
    fig2.savefig(f'{path}/accuracy_plot.png')

    with open(f"{path}/results.csv", "w") as f:
        f.write("EPOCHS,Training Loss,Validation Loss,Training Accuracy,Validation Accuracy\n")
        for i in range(len(epoch_list)):
            f.write(f"{epoch_list[i]},{train_loss_list[i]},{val_loss_list[i]}, {training_accuracy_list[i]},{validation_accuracy_list[i]}\n")

if __name__ == "__main__":
    app.run(main)