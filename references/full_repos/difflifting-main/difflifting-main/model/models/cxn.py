from model.models.base_model import BaseModel
from topomodelx.nn.cell.ccxn import CCXN as TopoCXN

class CXN(BaseModel):
    def __init__(self, in_channels, hidden_channels, out_channels, device, n_layers=2, **kwargs):
        super().__init__(in_channels, hidden_channels, out_channels)
        self.cxn = TopoCXN(in_channels, in_channels, in_channels, hidden_channels,device, **kwargs).to(device)

    def forward(self, x_0, x_1, adj_0, incidence_2):
        # Call the SCN2 model from TopoModelX

        return self.cxn.forward(x_0, x_1, adj_0, incidence_2.T) # dont know why but only works transposing incidence_1

    def __call__(self, *args, **kwargs):
        # Forward calls to the `forward` method
        return self.forward(*args, **kwargs)
