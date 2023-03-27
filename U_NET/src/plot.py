"""
Runs a model on a single node across multiple gpus.
"""
import os
# Enabling MPI support below 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from torchmetrics import MeanAbsolutePercentageError
import scipy.io as sio
import matplotlib.pyplot as plt
import configargparse

from src.LayoutDeepRegression import Model


def solidFilter(current):
    for j in range(64):
        for i in range(64):
            if current[i,j] >= 255 - 90:
                current[i,j] = 1
            else:
                current[i,j] = 0
    return current

def getAvgTemp(designField, tempField):
    designField = solidFilter(designField * 255)
    numSolidPixels = np.sum(designField)
    solidTemps = np.multiply(designField, tempField)
    totalSolidTemp = np.sum(solidTemps)
    avgTemp = totalSolidTemp / numSolidPixels
    return avgTemp

def main(hparams):
    model = Model(hparams).cuda()

    print(hparams)
    print()

    # Model loading
    model_path = os.path.join(f'lightning_logs/version_' +
                              hparams.test_check_num, 'checkpoints/')
    ckpt = list(Path(model_path).glob("*.ckpt"))[0]
    print(ckpt)

    model = model.load_from_checkpoint(str(ckpt), hparams = hparams)
    model.eval()
    model.cuda()
    mae_test = []

    # Testing Set
    root = hparams.data_root
    boundary = hparams.boundary
    test_list = hparams.test_list
    file_path = os.path.join(root, boundary, test_list)
    root_dir = os.path.join(root, boundary, 'test', 'test')

    count = 0
    avg = 0
    ree = [] 
    with open(file_path, 'r') as fp:
        for line in fp.readlines():
            # Incrementing count 
            count = count + 1
            # Data Reading
            data_path = line.strip()
            path = os.path.join(root_dir, data_path)
            data = sio.loadmat(path)
            u_true, layout = data["u"], data["F"]

            # Plot Layout and Real Temperature Field
            fig = plt.figure(figsize=(10.5, 3))

            grid_x = np.linspace(0, 0.1, num=64)
            grid_y = np.linspace(0, 0.1, num=64)
            X, Y = np.meshgrid(grid_x, grid_y)

            ax1 = plt.subplot(1, 3, 1)
            plt.title('Filtered Design', fontsize=12)
            # layoutTemp = np.rot90(layout, 2)
            oree = np.copy(layout)
            oree = oree * 255
            oree[oree < 90] = 0
            oree[oree >= 90] = 1
            im = plt.pcolormesh(X, Y, oree, cmap="Greys_r") 
            ax1.axis('off')
            cb1 = plt.colorbar(im)
            cb1.set_ticks([0.0,1.0])
            cb1.set_ticklabels(["0: Fluid", "1: Solid"])
            cb1.ax.tick_params(labelsize=12)
            fig.tight_layout(w_pad=0.05)

            design = np.copy(layout)
            layout = torch.Tensor(layout).unsqueeze(0).unsqueeze(0).cuda()
           
            print(layout.size())
            heat = torch.Tensor((u_true - 297) / 100.0).unsqueeze(0).unsqueeze(0).cuda()
            with torch.no_grad():
                heat_pre = model(layout)
                mae = F.l1_loss(heat, heat_pre) * 100
                print('MAE:', mae)
   
            mae_test.append(mae.item())
            heat_pre = heat_pre.squeeze(0).squeeze(0).cpu().numpy() * 100.0 + 297
            hmax = max(np.max(heat_pre), np.max(u_true))
            hmin = min(np.min(heat_pre), np.min(u_true))

            # Avg solid temperatures 
            pre_avg_solid_temp = round(getAvgTemp(design, heat_pre), 3)
            avg_solid_temp = round(getAvgTemp(design, u_true), 3)
            ree.append(abs(pre_avg_solid_temp-avg_solid_temp)/avg_solid_temp)
            print(f"AVG ERROR = {100 * np.mean(ree):.2f}%")

            ax2 = plt.subplot(1, 3, 2)
            plt.title('Simulated Temperature Field\n' + 'Avg solid temp: ' + str(pre_avg_solid_temp), fontsize=12)
            # u_true = np.rot90(u_true, 2)
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                im = plt.pcolormesh(xs, ys, u_true, vmin=hmin, vmax=hmax)
                plt.axis('equal')
            else:
                im = plt.pcolormesh(X, Y, u_true, vmin=hmin, vmax=hmax)
            ax2.axis('off')
            cbar = plt.colorbar(im)
            cbar.ax.locator_params(nbins=5)
            cbar.set_label("Temperature (K)")

            ax3 = plt.subplot(1, 3, 3)
            plt.title('Predicted Temperature Field\n' + 'Avg solid temp: ' + str(avg_solid_temp), fontsize=12)
            # heat_pre = np.rot90(heat_pre, 2)
            if "xs" and "ys" in data.keys():
                xs, ys = data["xs"], data["ys"]
                im = plt.pcolormesh(xs, ys, heat_pre, vmin=hmin, vmax=hmax)
                plt.axis('equal')
                plt.axis('off')
            else:
                im = plt.pcolormesh(X, Y, heat_pre, vmin=hmin, vmax=hmax)
            ax3.axis('off')
            cbar = plt.colorbar(im)
            cbar.ax.locator_params(nbins=5)
            cbar.set_label("Temperature (K)")


            save_name = os.path.join('outputs/predict_plot', os.path.splitext(os.path.basename(path))[0]+'.jpg')
            plt.tight_layout()
            fig.savefig(save_name, dpi=300)
            plt.close()

    mae_test = np.array(mae_test)
    print(mae_test.mean())
    np.savetxt('outputs/mae_test.csv', mae_test, fmt='%f', delimiter=',')


if __name__ == "__main__":

    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    # default configuration file
    config_path = Path(__file__).absolute().parent / "config/config.yml"
    parser = configargparse.ArgParser(default_config_files=[str(config_path)], description="Hyper-parameters.")

    # configuration file
    parser.add_argument("--config", is_config_file=True, default=False, help="config file path")

    # mode
    parser.add_argument("-m", "--mode", type=str, default="train", help="model: train or test or plot")

    # args for training
    parser.add_argument("--gpus", type=int, default=0, help="how many gpus")
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_epochs", default=20, type=int)
    parser.add_argument("--lr", default="0.01", type=float)
    parser.add_argument("--resume_from_checkpoint", type=str, help="resume from checkpoint")
    parser.add_argument("--num_workers", default=2, type=int, help="num_workers in DataLoader")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--use_16bit", type=bool, default=False, help="use 16bit precision")
    parser.add_argument("--profiler", action="store_true", help="use profiler")

    # args for validation
    parser.add_argument("--val_check_interval", type=float, default=1,
                        help="how often within one training epoch to check the validation set")

    # args for testing
    parser.add_argument("--test_check_num", default='0', type=str, help="checkpoint for test")
    parser.add_argument("--test_args", action="store_true", help="print args")

    parser = Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # test args in cli
    if hparams.test_args:
        print(hparams)
    else:
        main(hparams)
