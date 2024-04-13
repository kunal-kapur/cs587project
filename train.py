from torch.utils.data import dataloader
from books_dataloader import BooksDataset

data = BooksDataset(path="data/books_dataset.csv")

for cat, num, y in  data:
    
    print(cat)
    print(num)
    print(y)
    break