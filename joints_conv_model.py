class PointSolver_conv(nn.Module):
    def __init__(self, constant,in_channels=1, out_channels=1, kernel_size=3):
        super(PointSolver_conv, self).__init__()
        self.constant = constant
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = int(np.floor(self.kernel_size/2.))
        self.conv11 = nn.Conv1d(in_channels=self.in_channels, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.conv12 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.conv13 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.bn1 = nn.BatchNorm1d(num_features=16)
        
        self.conv21 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.conv22 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.conv23 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.bn2 = nn.BatchNorm1d(num_features=16)
        
        
        self.conv31 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.conv32 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.conv33 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        
        self.bn3 = nn.BatchNorm1d(num_features=16)
        
        self.conv41 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.conv42 = nn.Conv1d(in_channels=16, 
                               out_channels=16, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        self.conv43 = nn.Conv1d(in_channels=16, 
                               out_channels=self.out_channels, 
                               kernel_size=self.kernel_size, 
                               padding=self.padding)
        
        
    def forward(self, x):
        y = F.relu(self.conv11(self.constant+x))
        y = F.relu(self.conv12(y))
        y = F.relu(self.conv13(y)) 
        
        y = self.bn1(y)
        
        y = F.relu(self.conv21(y+x))
        y = F.relu(self.conv22(y))
        y = F.relu(self.conv23(y)) 
        
        y = self.bn2(y)
        
        y = F.relu(self.conv31(y+x))
        y = F.relu(self.conv32(y))
        y = F.relu(self.conv33(y))
        
        y = self.bn3(y)
        
        y = F.relu(self.conv41(y+x))
        y = F.relu(self.conv42(y))
        y = F.relu(self.conv43(y))
        
        return y