import torch
import config
from PIL import Image


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        y_gen_img = Image.fromarray(y_fake)
        y_gen_img.save(folder + f"/y_gen_{epoch}.png")
        inp_img =  Image.fromarray(x * 0.5 + 0.5)
        inp_img.save(folder + f"/input_{epoch}.png")
        if epoch == 1:
             lab_img = Image.fromarray(y * 0.5 + 0.5)
             lab_img.save(folder + f"/label_{epoch}.png")
    gen.train()
    
    
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    
def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr