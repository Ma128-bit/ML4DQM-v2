import pandas as pd
import numpy as np
import math, time, copy
import matplotlib.pyplot as plt

def Make_img(histo, Xbins, Xmin, Xmax, Ybins, Ymin, Ymax):
    img = np.zeros((100, 100), dtype=np.float32)
    
    for i in range(int(Ybins)):
        for j in range(int(Xbins)):
            img[i, j] = histo[i][j]#histo[i*(int(Xbins)+2)+j]
    #img = img[1:-1, 1:-1]
    return img
    
def Show2Dimg(img, title='CSC occupancy'):
    fig = plt.figure(figsize =(8, 8))
    img_temp = copy.deepcopy(img)
    cmap = plt.cm.jet
    cmap.set_under(color='white')
    max_=np.max(img_temp)
    img_temp[img_temp==0] = np.nan
    plt.imshow(img_temp, cmap=cmap, vmin=0.0000000001, vmax=max_)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()
    del img_temp
    #plt.savefig('CSC_occupancy.png')
    
def Show1Dimg(vx, vy, xfit=None, yfit=None, x=r"Lumi [10$^{33}$ cm$^{-2}$ s$^{-1}$]", y="Occupancy", eymin=4, eymax=4, marker='.', line=False, cms = False):
    if cms==True:
        import mplhep as hep
        hep.style.use("CMS")
        fig, ax = plt.subplots()
        hep.cms.label("Preliminary", data=True, ax=ax, loc=0, com=13.6, year=2024, lumi=61.5)
        ax.ticklabel_format(style="sci", scilimits=(-3, 3), useMathText=True)
        ax.get_yaxis().get_offset_text().set_position((-0.085, 1.05))
        hist, xedges, yedges = np.histogram2d(vx, vy, bins=100)
        hep.hist2dplot(hist, xedges, yedges, cmap='coolwarm', cmin=0.5, cmax=3500)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(vx, vy, marker=marker, linestyle='', markersize=5, label='Data')
    if line==True:
        ax.axvline(x=9, color='red', linestyle='--')
    if yfit is not None:
        ax.plot(xfit, yfit, color='red', label='Fit')
        ax.legend()
    
    #plt.ylim(0, 2000000)
    #plt.grid(True)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(eymin,eymax))
    if cms==False:
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_xlabel(x, fontsize='14')
        ax.set_ylabel(y, fontsize='14')
        ax.set_title(r'$\mathbf{CMS}\ \mathit{Private\ work}$', x=0.24, y=1.0, fontsize=14)
        #plt.title('CMS', fontweight='bold',x=0.12, y=1.0, size=14)
        ax.set_title('2024 (13.6 TeV)',loc='right', fontsize=14)
        ax.legend()
    else:
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    plt.show()
    
def PLots_in_training(X, Xreco):
    print(' >> original image:')
    img = X[0][0].cpu().numpy()
    Show2Dimg(img)
    print(' >> AE-reco image:')
    img_reco = Xreco[0][0].detach().cpu().numpy()
    Show2Dimg(img_reco)
    print(' >> loss map:')
    img_loss = F.mse_loss(Xreco[0], X[0], reduction='none')[0].detach().cpu().numpy()
    Show2Dimg(img_loss)

def Show2DLoss(img, vmin=-1.5, vmax=2., title='Loss'):
    fig = plt.figure(figsize =(8, 8))
    img_temp = copy.deepcopy(img)
    cmap = plt.cm.seismic
    cmap.set_bad(color='black')
    img_temp[img_temp==0] = np.nan
    plt.imshow(img_temp, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()
    del img_temp

def entry_plot(entry):
    mean_entries_norm = np.mean(entry)
    std_entries_norm = np.std(entry)
    up = mean_entries_norm+1*std_entries_norm
    down = mean_entries_norm-1*std_entries_norm
    
    plt.figure(figsize=(7, 4))
    plt.plot(entry, label='entries_norm', alpha=0.7)
    
    plt.axhline(y=mean_entries_norm, color='r', linestyle='--', linewidth=2, label='Mean')
    
    plt.fill_between(
        range(len(entry)),
        down,  
        up,  
        color='red',
        alpha=0.5,  
        label='±1 STD'  
    )
    
    plt.fill_between(
        range(len(entry)),
        up,  
        up+std_entries_norm,  
        color='orange',
        alpha=0.5,  
        label='±2 STD'  
    )
    plt.fill_between(
        range(len(entry)),
        down-std_entries_norm,  
        down,
        color='orange',
        alpha=0.5,
    )
    
    plt.fill_between(
        range(len(entry)),
        up+std_entries_norm,  
        up+2*std_entries_norm,  
        color='yellow',
        alpha=0.5,  
        label='±3 STD'  
    )
    plt.fill_between(
        range(len(entry)),
        down-2*std_entries_norm,  
        down-std_entries_norm,
        color='yellow',
        alpha=0.5,
    )
    
    plt.legend()
    
    plt.show()
