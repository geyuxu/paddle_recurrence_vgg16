import os
import paddle

def save_model(model, optimizer, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    paddle.save(model.state_dict(), f"{save_dir}/vgg16_epoch{epoch}.pdparams")
    paddle.save(optimizer.state_dict(), f"{save_dir}/vgg16_epoch{epoch}.pdopt")

def load_model(model, path):
    state_dict = paddle.load(path)
    model.set_state_dict(state_dict)
    return model
