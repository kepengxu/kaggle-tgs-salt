from utils.DataPare import provider
import settings
import random
from tqdm import tqdm
random.seed(2019)
import matplotlib.pyplot as plt
# data_folder=settings.data_folder
df_path=settings.df_path
dataloader = provider(
    fold=1,
    total_folds=5,
    # data_folder=data_folder,
    df_path=df_path,
    phase="train",
    size=512,
    mean = (0.459),
    std = (0.224*2),
    batch_size=16,
    num_workers=1,
)

for i in tqdm(range(10000)):
    batch = next(iter(dataloader)) # get a batch from the dataloader
    images, masks = batch
    # print(images.shape,masks.shape)

# # plot some random images in the `batch`
# idx = random.choice(range(16))
# plt.imshow(images[idx][0], cmap='bone')
# plt.imshow(masks[idx][0], alpha=0.2, cmap='Reds')
# plt.show()

