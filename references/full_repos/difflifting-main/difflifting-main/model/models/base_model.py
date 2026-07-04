from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, in_channels, hidden_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x_0, x_1, x_2, lap_0, lap_1, lap_2):
        pass
