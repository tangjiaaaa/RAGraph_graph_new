from model.models.cwn import CWN
from model.models.scn2 import SCN2
from model.models.cxn import CXN

class ModelFactory:
    @staticmethod
    def create_model(model_type, in_channels, hidden_channels, out_channels, device, **kwargs):
        if model_type == "CWN":
            return CWN(in_channels, hidden_channels, out_channels, device,**kwargs)
        elif model_type == "SCN2":
            return SCN2(in_channels, hidden_channels, out_channels,device, **kwargs)
        elif model_type == "CXN":
            return CXN(in_channels, hidden_channels, out_channels, device,**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
