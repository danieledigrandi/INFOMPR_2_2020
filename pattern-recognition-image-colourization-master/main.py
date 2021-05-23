from dataset import ImageDataset as IDs
from model import MODEL as mod


network = mod()
network.build()

network.train(IDs(p=0.01))
network.test(IDs(p=0.001))
