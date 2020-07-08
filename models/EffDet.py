import sys
from config import *
sys.path.insert(0, timm_efficientdet_path)
sys.path.insert(0, omegaconf_path)
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet

def EffDet(pretrained_model='tf_efficientdet_d5', img_size=512):
    config = get_efficientdet_config(pretrained_model)
    net = EfficientDet(config, pretrained_backbone=True)
    # checkpoint = torch.load('../input/efficientdet/efficientdet_d5-ef44aea8.pth')
    # net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = img_size
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return DetBenchTrain(net, config)
