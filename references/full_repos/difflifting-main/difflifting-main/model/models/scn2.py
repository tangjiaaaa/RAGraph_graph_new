from model.models.base_model import BaseModel
from topomodelx.nn.simplicial.scn2 import SCN2 as TopoSCN2

class SCN2(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels, device ,n_layers=2, **kwargs):
        super().__init__(in_channels, hidden_channels, out_channels)
        print(device)
        self.scn2 = TopoSCN2(in_channels, hidden_channels, out_channels, n_layers=n_layers, **kwargs).to(device)

    def forward(self, x_0, x_1, x_2, lap_0, lap_1, lap_2):
        # Call the SCN2 model from TopoModelX
        return self.scn2.forward(x_0, x_1, x_2, lap_0, lap_1, lap_2)

    def __call__(self, *args, **kwargs):
        # Forward calls to the `forward` method
        return self.forward(*args, **kwargs)
