from models.backbones.vgg import VGG
from models.backbones.utils import kaiming_init, normal_init, constant_init


class SSDVGG(VGG):
    extra_setting = {
        300: (256, "S", 512, 128, "S", 256, 128, 256, 128, 256),
        512: (256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256, 128)
    }

    def __init__(self,
                 input_size,
                 depth,
                 with_last_pool=False,
                 ceil_mode=True,
                 out_indices=(3, 4),
                 out_feature_indices=(22, 34),
                 l2_norm_scale=20.):
        super(SSDVGG, self).__init__(
            depth=depth,
            with_last_pool=with_last_pool,
            ceil_mode=ceil_mode,
            out_indices=out_indices
        )
        assert input_size in (300, 512)
        self.input_size = input_size

