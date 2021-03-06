from .unet_parts import *


class UNet(nn.Module):
    """
    普通实现的Unet
    :n_channels 输入图片的channel数
    :o_channels 输出图片的channel数
    """

    def __init__(self, n_channels, o_channels):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, o_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)


class WTUNet(nn.Module):
    """
    使用DWT,IWT的Unet模型
    :n_channels 输入图片的channel数
    :o_channels 输出图片的channel数
    """

    def __init__(self, n_channels, o_channels):
        super(WTUNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down_with_dwt(64 * 4, 128)
        self.down2 = down_with_dwt(128 * 4, 256)
        self.down3 = down_with_dwt(256 * 4, 512)
        self.down4 = down_with_dwt(512 * 4, 512)
        self.up1 = up_with_iwt(128 + 512, 256)
        self.up2 = up_with_iwt(64 + 256, 128)
        self.up3 = up_with_iwt(32 + 128, 64)
        self.up4 = up_with_iwt(16 + 64, 64)
        self.outc = outconv(64, o_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)
