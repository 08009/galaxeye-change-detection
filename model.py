
import segmentation_models_pytorch as smp

def build_model(encoder="resnet34", encoder_weights="imagenet"):
    """
    U-Net with ResNet34 backbone.
    in_channels=4 because input is 3-band EO + 1-band SAR concatenated.
    classes=1 for binary change map output.
    """
    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=4,   # 3 EO + 1 SAR
        classes=1,       # binary output
        activation=None, # raw logits — sigmoid applied manually
    )
    return model
