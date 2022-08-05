import optuna
from torch import nn
import torch
import torch.optim as optim
from kornia.losses import focal # pip3 install kornia will be required
import numpy as np
from sklearn import metrics
from pathlib import Path
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import optuna # requires pip3 install optuna

# Load the already computed downsampled dataset

downsample_size = 10#50 # the biggest it is, the smaller the brain 3D snapshots are.

data_base = Path("/home/jovyan/shared/NSD")
curr_subj = "subj01"
subj_behav_file = data_base / f"nsddata/ppdata/{curr_subj}/behav/responses.tsv"
subj_behav = pd.read_csv(subj_behav_file, delimiter="\t")

# add within session iteration indices
for sess_id in range(1, 37+1):
    subj_behav.loc[subj_behav["SESSION"]==sess_id, "INSESSIONIDX"] = list(range(1, 750+1))
    

subj_behav_sel = subj_behav.dropna(subset=["BUTTON"])
subj_behav_sel = subj_behav_sel[~subj_behav_sel["SESSION"].isin([38, 39, 40])]


# take the first saved data
betas = np.load("./nsd-data/MTL_downsampled_"+str(downsample_size)+"_full.npy")
betas = betas.reshape((betas.shape[3],betas.shape[0],betas.shape[1],betas.shape[2] ))

X = betas[subj_behav_sel.index, :] / 300
y = subj_behav_sel["ISOLDCURRENT"].to_numpy()

#print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

input_dims = [X.shape[1],X.shape[2],X.shape[3]]
train_dataset_size = X_train.shape[0]
test_dataset_size = X_test.shape[0]
train_dataset_ids = [i for i in range(train_dataset_size)]
minibatch_size = 32


def objective(trial):
        # ======= model creation

    #torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

    model = nn.Sequential(
        nn.Conv3d(1,8, 5), #input_nb_channels=1, output_nb_channels=1, kernel size
        nn.SELU(),
        nn.Conv3d(8,16,3),
        nn.SELU(),
        nn.Dropout(p=0.5),
        nn.Conv3d(16,32,3),
        nn.SELU(),
        nn.Dropout(p=0.5),
        nn.Flatten(),
        # resampled 25 version
        nn.Linear(96, 2), # Output size of the previous layer, output size = 1 (probability)
        # all 125 version
        # nn.Linear(238, 2), # Output size of the previous layer, output size = 1 (probability)
        #nn.Dropout(),
        nn.Softmax(dim=1)
        # nn.Linear(238, 1), # Output size of the previous layer, output size = 1 (probability)
        # nn.Sigmoid() # proba
    )


    #========== hyperparameters (optuna)

    #lr, w_decay = 3e-4, 0.0 # TODO change
    #gamma = 30#30#100#5#100.0#10.0 any gamma value seems to work /!\ the biggest gamma is, the more weight is given to less present data (ie true positive)
    # 5 is too small (no true positive); but 100 is too high (not enough true negatives)
    
    w_decay = 0.0
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-1)
    #lr = trial.suggest_uniform('lr', 1e-4, 1e-1)
    gamma = trial.suggest_uniform('gamma', 5.0, 40.0)
    
    alpha = 1.0#1.0#1.0 works#0.5 works #0.0 does not work
    '''
    alpha: Weighting factor :math:`\alpha \in [0, 1]`.
            gamma: Focusing parameter :math:`\gamma >= 0`.
    '''

    # ==== optimizer & loss function
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay= w_decay)
    # Loss = nn.BCEWithLogitsLoss()
    
    
    #======= training loop
    epochs = 2000
    #epochs = 10000
    # epochs = 1000
    #epochs = 200
    #pbar = tqdm(range(epochs))
    for epoch in range(epochs):
        #pbar.set_description(str(epoch))

        # 1) fetch minibatch
        minibatch_ids = np.random.choice(train_dataset_ids, size=minibatch_size)

        #minibatch_list = []
        #minibatch_labels_list = []
        #for id in minibatch_ids:
            #minibatch_list.append(dataset['X'][id])
            #minibatch_labels_list.append(dataset['y'][id])


        minibatch_X = X_train[minibatch_ids, :,:,:]
        minibatch_y = y_train[minibatch_ids]


        minibatch_tensor_input = torch.tensor(minibatch_X ).float().reshape(minibatch_size, 1, input_dims[0], input_dims[1], input_dims[2]) # the additional 1 = number of input channels
        minibatch_tensor_labels = torch.tensor(minibatch_y).reshape(minibatch_size, 1)

        # print(minibatch_tensor_labels.size())
        # print(minibatch_tensor_labels)

        # 2) get outputs probabilities (target, non target)
        minibatch_tensor_output = model(minibatch_tensor_input)
        #print(minibatch_tensor_output)
        # print(minibatch_tensor_output.size())

        #3) compute loss function
        # print(minibatch_tensor_output.reshape(minibatch_size), minibatch_tensor_labels.reshape(minibatch_size))
        # print(minibatch_tensor_output, minibatch_tensor_labels)
        optimizer.zero_grad()


        loss = focal.focal_loss(minibatch_tensor_output, minibatch_tensor_labels.reshape(minibatch_size), alpha=alpha, gamma=gamma).mean() # works
        # print(loss)
        #losses.append(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-2) # avoid exploding gradients
        optimizer.step()


        
        
    # =============== END OF TRAINING PERFORMANCES = see performances on testing set TODO adapt
    # 1) fetch test data
    test_tensor_input = torch.tensor(X_test).float().reshape(test_dataset_size, 1, input_dims[0], input_dims[1],
                                                                 input_dims[2])
    test_tensor_labels = torch.tensor(y_test).reshape(test_dataset_size, 1)


    # 2) get outputs probabilities (target, non target)
    model.eval() # disable dropout for test time
    with torch.no_grad():
        test_tensor_output = model(test_tensor_input)

    # === compute scores TODO remove !!! OR SET IT ON A TEST SET !!!
    y_pred = torch.argmax(test_tensor_output, dim=1)


    # --- Plot ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(test_tensor_labels.numpy().flatten(),y_pred.numpy(), pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
    #display.plot()
    #plt.show()
    #print("--> roc_auc: ", roc_auc)
    return roc_auc



