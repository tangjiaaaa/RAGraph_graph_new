from model.models.base_model import BaseModel
from topomodelx.nn.cell.cwn import CWN as TopoCWN

class CWN(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels, device, n_layers=2,  **kwargs):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.cwn = TopoCWN(in_channels, in_channels, in_channels, hidden_channels, n_layers=n_layers, **kwargs).to(device)

    def forward(self, x_0, x_1, x_2, adj_0, incidence_2, incidence_1):
        # Call the SCN2 model from TopoModelX
        return self.cwn.forward(x_0, x_1, x_2, adj_0, incidence_2, incidence_1.T) # dont know why but only works transposing incidence_1

    def __call__(self, *args, **kwargs):
        # Forward calls to the `forward` method
        return self.forward(*args, **kwargs)
