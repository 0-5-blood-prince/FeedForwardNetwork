import wandb
import dataset
wandb.login(key="866040d7d81f67025d43e7d50ecd83d54b6cf977", relogin=False)

def Q1():
    wandb.init(project="feedforwardfashion")
    dataset.log_images()
def confusion_matrix():
    pass
Q1()