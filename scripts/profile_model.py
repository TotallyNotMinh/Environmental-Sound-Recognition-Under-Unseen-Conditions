import torch
from torchinfo import summary
from models.classifier import Classifer

for enc in ["resnet34", "resnet152"]:
    model = Classifer(encoder_type=enc)
    print(f"\n================ Profiling {enc} ================")
    # Input tensor giả định: (batch_size=1, channels=1, freq=128, time=500)
    summary(model, input_size=(1, 1, 128, 500), col_names=["num_params", "mult_adds"])