
import torch
import torch.nn as nn
import torch.optim as optim

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=1e-4)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, biggest_layer=1024):
        super(Generator, self).__init__()

        # Contracting Path (Encoder)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, biggest_layer // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(biggest_layer // 2, biggest_layer // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(biggest_layer // 2, biggest_layer, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(biggest_layer, biggest_layer, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.drop5 = nn.Dropout(0.5)

        # Expanding Path (Decoder)
        self.up6 = nn.ConvTranspose2d(biggest_layer, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(biggest_layer // 2 + 512, 512, kernel_size=3, padding=1), # Concatenation channel adjustment
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1), # Concatenation channel adjustment
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), # Concatenation channel adjustment
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), # Concatenation channel adjustment
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1), # Keras had 3x3 then 1x1, here combined
            nn.ReLU(inplace=True) # Assuming the 2-channel conv was an intermediate step, not the final activation
        )

        self.conv10 = nn.Conv2d(64, output_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Apply He normal initialization
        self.apply(init_weights)

    def forward(self, x):
        # Contracting Path
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)

        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)

        # Expanding Path
        up6 = self.up6(drop5)
        # Check for size mismatch and crop if necessary
        if up6.shape[2:] != drop4.shape[2:]:
            up6 = nn.functional.interpolate(up6, size=drop4.shape[2:], mode='bilinear', align_corners=False)
        merge6 = torch.cat([drop4, up6], dim=1) # Concatenate along channel dimension
        conv6 = self.conv6(merge6)

        up7 = self.up7(conv6)
        if up7.shape[2:] != conv3.shape[2:]:
            up7 = nn.functional.interpolate(up7, size=conv3.shape[2:], mode='bilinear', align_corners=False)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)

        up8 = self.up8(conv7)
        if up8.shape[2:] != conv2.shape[2:]:
            up8 = nn.functional.interpolate(up8, size=conv2.shape[2:], mode='bilinear', align_corners=False)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)

        up9 = self.up9(conv8)
        if up9.shape[2:] != conv1.shape[2:]:
            up9 = nn.functional.interpolate(up9, size=conv1.shape[2:], mode='bilinear', align_corners=False)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9(merge9)

        conv10 = self.conv10(conv9)
        output = self.sigmoid(conv10)
        return output


class Discriminator(nn.Module):
    def __init__(self, input_channels=1):
        super(Discriminator, self).__init__()
        df = 64

        def d_layer(in_filters, out_filters, f_size=4, bn=True):
            layers = [
                nn.Conv2d(in_filters, out_filters, kernel_size=f_size, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if bn:
                layers.append(nn.BatchNorm2d(out_filters, momentum=0.8))
            return nn.Sequential(*layers)

        # The discriminator receives concatenated images, so input channels are doubled
        self.d1 = d_layer(input_channels * 2, df, bn=False)
        self.d2 = d_layer(df, df * 2)
        self.d3 = d_layer(df * 2, df * 4)
        self.d4 = d_layer(df * 4, df * 4) # Original had df*4 again, so it's a 256->256 transition

        # Output layer
        self.validity = nn.Conv2d(df * 4, 1, kernel_size=4, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

        # Apply He normal initialization
        self.apply(init_weights)

    def forward(self, img_A, img_B):
        combined_imgs = torch.cat([img_A, img_B], dim=1)

        d1 = self.d1(combined_imgs)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        validity = self.validity(d4)
        validity = self.sigmoid(validity)
        return validity

class GAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, input_img):
        self.discriminator.eval()

        generated_img = self.generator(input_img)
        validity = self.discriminator(generated_img, input_img)

        self.discriminator.train()
        
        return validity, generated_img
