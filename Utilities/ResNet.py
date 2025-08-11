import pandas as pd
import numpy as np
import torch, time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(df: pd.DataFrame, output_path: str, data_name: str, batch_size: int = 64, epochs: int = 50, learning_rate: float = 2.e-4, patience: int = 4, threshold: float = 0.005) -> torch.jit.ScriptModule:
    #df = df[df["good_flag"]==0]
    df[data_name] = df[data_name].apply(lambda histo: np.vstack(histo).astype(np.float64))
    #df[data_name] = df[data_name].apply(lambda histo: np.array(histo, dtype=np.float64))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    while True:
        trainer = ModelTrainer(df, data_name)
        if trainer.train() != -1:
            break
        del trainer
    scripted_model = torch.jit.script(trainer.model)

    scripted_model.save(f"{output_path}/trained_model_{data_name}_{device}.pth")
    if torch.cuda.is_available():
        scripted_model.to("cpu").save(f"{output_path}/trained_model_{data_name}_cpu.pth")
        scripted_model.to(device)

    return scripted_model

def predictions(df: pd.DataFrame, model: torch.jit.ScriptModule, data_name: str, num_sectors: int) -> pd.DataFrame:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    selected_chamber = df[data_name]
    tensor = torch.stack([torch.tensor(m, dtype=torch.float32).unsqueeze(0) for m in selected_chamber])
    loader = DataLoader(dataset=tensor, batch_size=128, num_workers=1, shuffle=False)

    reco_imgs, loss_imgs = [], []

    model.eval()
    with torch.no_grad():
        for img_batch in loader:
            img_batch = img_batch.to(device)
            # Computing Model predinctions: model(img)--> reco_img 
            reco_img_batch = model(img_batch)
            # Computing Loss as (img-reco_img)/reco_img
            img_loss_batch = (img_batch - reco_img_batch)[:, 0] / reco_img_batch[:, 0]
            loss_imgs.extend(img_loss_batch.cpu().numpy())
            reco_imgs.extend(reco_img_batch[:, 0].cpu().numpy())

    loss_imgs = [np.where(np.isinf(matrix), 2, matrix) for matrix in loss_imgs]
    df["reco_img"] = reco_imgs
    df["loss_img"] = loss_imgs
    df["real_flag"] = 1

    binary_matrix = (np.mean(df[data_name], axis=0) != 0)
    if num_sectors is None:
        df["max_loss"] = df["loss_img"].apply(np.nanmax)
        df["min_loss"] = df["loss_img"].apply(np.nanmin)
    else:
        from Utilities.CSCRebinning import rebin_image
        df["rebinned_loss_img"] = df.apply( lambda row: rebin_image(row["loss_img"], binary_matrix, num_sectors), axis=1)
        df["max_loss"] = df["rebinned_loss_img"].apply(np.nanmax)
        df["min_loss"] = df["rebinned_loss_img"].apply(np.nanmin)
        df["rebinned_loss_img"] = df["rebinned_loss_img"].apply(lambda x: x.tolist())
    
    df[data_name] = df[data_name].apply(lambda x: x.tolist())
    df["reco_img"] = df["reco_img"].apply(lambda x: x.tolist())
    df["loss_img"] = df["loss_img"].apply(lambda x: x.tolist())
    return df

class ModelTrainer:
    def __init__(self, df, data_name, batch_size=64, epochs=50, learning_rate=2.e-4, patience=4, threshold=0.005):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.threshold = threshold
        self.debug = False
        
        # Caricamento e preparazione dei dati
        self.training_loader, self.validation_loader, self.img_size = self._load_data(df, data_name)
        
        # Inizializzazione del modello
        self.model = ResNetAE(1, 3, [16, 32, 64], img_size=self.img_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=2, threshold=0.02, verbose=True)
        
        self.train_loss = []
        self.val_loss = []
        self.min_val_loss = 1e6
        self.best_model_state = None
        self.no_improve_epochs = 0

    def _load_data(self, data, data_name):
        # Caricamento e preprocessamento dei dati
        #data = np.load(data_path, allow_pickle=True)
        selected_chamber = data[data_name]
        tensor_list = [(torch.tensor(m, dtype=torch.float32)).unsqueeze(0) for m in selected_chamber]
        if self.debug:
            print(tensor_list[0].size())
        
        dim = len(tensor_list)
        training_dim = int(dim * 0.85)
        np.random.seed(42)  
        indices = np.random.permutation(len(tensor_list)) 
        training_tensor = [tensor_list[i] for i in indices[:training_dim]]
        validation_tensor = [tensor_list[i] for i in indices[training_dim:]]
        print(f"Data ({len(selected_chamber)}) = Training sample ({len(training_tensor)}) + Validation sample ({len(validation_tensor)})")

        training_loader = DataLoader(dataset=training_tensor, batch_size=self.batch_size, num_workers=1, shuffle=False)
        validation_loader = DataLoader(dataset=validation_tensor, batch_size=self.batch_size, num_workers=1, shuffle=False)

        # Determinazione della dimensione dell'immagine
        img_size = list(tensor_list[0].squeeze(0).size())
        
        return training_loader, validation_loader, img_size

    def train(self):
        for epoch in range(self.epochs):
            print(f'***** Training Epoch {epoch + 1} *****')
            self.model.train()
            train_loss = self._train_epoch(epoch)
            if ((epoch==0) & (train_loss == -1)):
                print("No improvements in the first epoch. Exiting.")
                return -1
            self.train_loss.append(train_loss)
            
            print(f"Evaluating on validation set")
            self.model.eval()
            val_loss = self._validate_epoch()
            self.val_loss.append(val_loss)
            
            self.scheduler.step(val_loss)
            self._early_stopping(val_loss)
            
            if self.no_improve_epochs >= self.patience:
                print("Early stopping triggered.")
                return 0
        return 0
        
    def _train_epoch(self, epoch):
        tloss = []
        start_time = time.time()

        initial_loss = 100
        final_loss = 0

        for i, figure in enumerate(self.training_loader):
            X = figure.to(self.device)
            self.optimizer.zero_grad()
            Xreco = self.model(X)
            
            loss = F.l1_loss(Xreco, X)
            tloss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            
            if i % 70 == 0:
                print(f'>> [{i}/{len(self.training_loader)}] Train loss: {loss.item()}')
            
            if (epoch==0) & (i==0):
                initial_loss = loss.item()
            if (epoch==0) & (i==self.training_loader.__len__()-1):
                final_loss = np.mean(tloss)

        if ((epoch == 0) & ((initial_loss-final_loss)/initial_loss < 0.2)):
            print(f"Initial loss: {initial_loss}")
            print(f"Final loss: {final_loss}")       
            return -1 
            
        epoch_time = time.time() - start_time
        print(f"Training time: {epoch_time / 60:.2f} min in {len(self.training_loader)} steps")
        return np.mean(tloss)

    def _validate_epoch(self):
        vloss = []
        with torch.no_grad():
            for i, figure in enumerate(self.validation_loader):
                X_val = figure.to(self.device)
                Xreco_val = self.model(X_val)
                loss_val = F.l1_loss(Xreco_val, X_val)
                vloss.append(loss_val.item())
        
        avg_val_loss = np.mean(vloss)
        print(f'>> Validation loss: {avg_val_loss}')
        return avg_val_loss

    def _early_stopping(self, avg_val_loss):
        print(f"Current minimum: {self.min_val_loss}, relative diff: {(self.min_val_loss - avg_val_loss) / self.min_val_loss}")

        if (self.min_val_loss - avg_val_loss) / self.min_val_loss < self.threshold:
            self.no_improve_epochs += 1
        else:
            self.min_val_loss = avg_val_loss
            self.best_model_state = self.model.state_dict()
            self.no_improve_epochs = 0
        print(f"No improve epochs: {self.no_improve_epochs}")

class ResNetAE(nn.Module):
    """
    Define the full ResNet autoencoder model
    """

    def __init__(self, in_channels, nblocks, fmaps, img_size):
        super(ResNetAE, self).__init__()

        self.fmaps = fmaps
        self.nblocks = nblocks
        self.in_channels = in_channels
        self.img_size = img_size

        self.debug = False
        self._initialize_encoding_layers(in_channels, fmaps)
        self._initialize_sizes(img_size)
        self._initialize_decoding_layers(in_channels, fmaps)

    def _initialize_encoding_layers(self, in_channels, fmaps):
        # Initialize encoding layers
        self.econv0 = nn.Sequential(
            nn.Conv2d(in_channels, fmaps[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.elayer1 = self.block_layers(self.nblocks, [fmaps[0], fmaps[0]], "enc")
        self.elayer2 = self.block_layers(1, [fmaps[0], fmaps[1]], "enc")
        self.elayer3 = self.block_layers(self.nblocks, [fmaps[1], fmaps[1]], "enc")
        self.elayer4 = self.block_layers(1, [fmaps[1], fmaps[2]], "enc")
        self.elayer5 = self.block_layers(self.nblocks, [fmaps[2], fmaps[2]], "enc")

    def _initialize_sizes(self, img_size):
        # initialize the sizes across the layers
        self.size0 = list(img_size)
        self.size2 = [int(np.floor(el * 0.5)) for el in self.size0]
        self.size4 = [int(np.ceil(el * 0.5)) for el in self.size2]
        self.size5 = [int(np.ceil(el * 0.5)) for el in self.size4]
        if self.debug:
            print(f"size0: {self.size0}")
            print(f"size2: {self.size2}")
            print(f"size4: {self.size4}")
            print(f"size5: {self.size5}")

    def _initialize_decoding_layers(self, in_channels, fmaps):
        # Initialize decoding layers
        self.fc = nn.Linear(
            self.fmaps[-1], self.fmaps[-1] * self.size5[0] * self.size5[1]
        )  # 5x5
        self.dlayer5 = self.block_layers(
            self.nblocks, [fmaps[2], fmaps[2]], "dec", out_shape=None
        )
        self.dlayer4 = self.block_layers(
            1, [fmaps[2], fmaps[1]], "dec", out_shape=self.size4
        )
        self.dlayer3 = self.block_layers(
            self.nblocks, [fmaps[1], fmaps[1]], "dec", out_shape=None
        )
        self.dlayer2 = self.block_layers(
            1, [fmaps[1], fmaps[0]], "dec", out_shape=self.size2
        )
        self.dlayer1 = self.block_layers(
            self.nblocks, [fmaps[0], fmaps[0]], "dec", out_shape=None
        )
        self.dconv0 = nn.ConvTranspose2d(
            fmaps[0], in_channels, kernel_size=3, stride=1, padding=(1, 1)
        )
        self.dconv0_relu = nn.ReLU(inplace=True)

    def block_layers(self, nblocks, fmaps, state, out_shape=None):
        """
        Convenience function: append several resnet blocks in sequence
        """
        layers = []
        for _ in range(nblocks):
            if state == "enc":
                layers.append(ResBlock(fmaps[0], fmaps[1]))
            else:
                layers.append(ResBlockTranspose(fmaps[0], fmaps[1], out_shape))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.encode(x)
        x = self.decode(x)

        return x

    def encode(self, x):
        if self.debug:
            print(x.size())
        if self.debug:
            print("Encode")
        x = self.econv0(x)
        if self.debug:
            print(x.size())
        x = F.max_pool2d(x, kernel_size=2)
        if self.debug:
            print(x.size())

        x = self.elayer1(x)
        if self.debug:
            print(x.size())
        x = self.elayer2(x)
        if self.debug:
            print(x.size())
        x = self.elayer3(x)
        if self.debug:
            print(x.size())
        x = self.elayer4(x)
        if self.debug:
            print(x.size())
        x = self.elayer5(x)
        if self.debug:
            print(x.size())

        # Bottleneck comes from GlobalMaxPool
        if self.debug:
            print("Maxpool-FC")
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        if self.debug:
            print(x.size())
        x = x.view(x.size()[0], -1)
        if self.debug:
            print(x.size())

        return x

    def decode(self, x):
        # Expand bottleneck
        # Dimensions follow encoding steps in reverse, as much as possible
        x = self.fc(x)  # expand
        if self.debug:
            print(x.size())
        x = x.view(-1, self.fmaps[-1], self.size5[0], self.size5[1])
        if self.debug:
            print(x.size())

        # Decoding
        if self.debug:
            print("Decode")
        x = self.dlayer5(x)
        if self.debug:
            print(x.size())
        x = self.dlayer4(x)
        if self.debug:
            print(x.size())
        x = self.dlayer3(x)
        if self.debug:
            print(x.size())
        x = self.dlayer2(x)
        if self.debug:
            print(x.size())
        x = self.dlayer1(x)
        if self.debug:
            print(x.size())

        x = F.interpolate(x, size=list(self.size0))
        if self.debug:
            print(x.size())
        x = self.dconv0(
            x, output_size=(x.size()[0], self.in_channels, self.size0[0], self.size0[1])
        )
        if self.debug:
            print(x.size())
        x = self.dconv0_relu(x)
        if self.debug:
            print(x.size())

        return x


class ResBlock(nn.Module):
    """
    For encoding, define the nominal resnet block
    """

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = out_channels // in_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=self.downsample
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample > 1:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)

        return out


class ResBlockTranspose(nn.Module):
    """
    For decoding, define the transposed resnet block, aka "de-convolution"
    """

    def __init__(self, in_channels, out_channels, out_shape=None):
        super(ResBlockTranspose, self).__init__()
        self.upsample = in_channels // out_channels
        self.convT1 = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=self.upsample, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.convT2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=3, padding=1
        )

        self.shortcutT = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=1, stride=self.upsample
        )
        self.out_shape = out_shape

    def forward(self, x):
        residual = x

        if self.out_shape is None:
            out = self.convT1(x)
        else:
            out = self.convT1(
                x,
                output_size=(
                    x.size()[0],
                    x.size()[1],
                    self.out_shape[0],
                    self.out_shape[1],
                ),
            )
        out = self.relu(out)
        out = self.convT2(out)

        if self.upsample > 1:
            if self.out_shape is None:
                residual = self.shortcutT(x)
            else:
                residual = self.shortcutT(
                    x,
                    output_size=(
                        x.size()[0],
                        x.size()[1],
                        self.out_shape[0],
                        self.out_shape[1],
                    ),
                )

        out += residual
        out = self.relu(out)

        return out
