import paddle
import paddle.nn as nn

def make_vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2D(in_channels, out_channels, 3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2D(2, 2))
    return nn.Sequential(*layers)

class VGG16(nn.Layer):
    def __init__(self, num_classes=1000, global_pool=True):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(2, 3, 64),
            make_vgg_block(2, 64, 128),
            make_vgg_block(3, 128, 256),
            make_vgg_block(3, 256, 512),
            make_vgg_block(3, 512, 512),
        )
        self.global_pool = global_pool
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = paddle.mean(x, axis=[2, 3]) if self.global_pool else paddle.flatten(x, 1)
        return self.classifier(x)
