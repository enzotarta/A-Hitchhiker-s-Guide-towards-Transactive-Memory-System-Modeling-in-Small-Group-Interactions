import torch
import torch.optim as optim
import torch.backends.cudnn

import os
import torchvision
from torchvision import transforms

import json
import os
import numpy as np

from dataloader import *
import sklearn
from sklearn import tree

from dataloader import WoNoWa_DataLoader
from sklearn.tree import export_graphviz
from scipy.cluster.vq import vq, kmeans, whiten

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

def BA_compute(expected, y_test, this_id):
    this_BA = np.zeros(5)
    for i in range(5):
        if len(y_test.shape) == 1:
            this_y_test = (y_test== i)
        else:
            this_y_test = (y_test[:,this_id].numpy() == i)
        #print(this_y_test)
        total = np.sum(this_y_test)
        this_predicted = (expected == i).numpy()
        #print(this_predicted)
        predicted = np.sum(this_predicted)
        if (total == 0) and (predicted == 0):
            this_BA[i] = 1.0
        elif (total == 0) and (predicted != 0):
            this_BA[i] = 0.0
        else:
            true_positives = np.mean((this_y_test == True) * (this_predicted == True))
            true_negatives = np.mean((this_y_test == False) * (this_predicted == False))
            false_positives = np.mean((this_y_test == False) * (this_predicted == True))
            false_negatives = np.mean((this_y_test == True) * (this_predicted == False))
            sensitivity = 1.0 * true_positives / (true_positives + false_negatives + 0.000000001)
            specificity = 1.0 * true_negatives / (true_negatives + false_positives + 0.000000001)
            this_BA[i] = 1.0 * (sensitivity + specificity) / 2
        #print(i, this_BA[i], this_y_test == this_predicted)
    return np.mean(this_BA[2:])

# function for fitting trees of various depths on the training data using cross-validation
def run_cross_validation_on_trees(X, y, tree_depths, scoring='balanced_accuracy'):
    cv_scores_list = []
    cv_scores_std = []
    cv_scores_mean = []
    accuracy_scores = []
    for depth in tree_depths:
        this_scores_mean=[]
        this_scores_std=[]
        this_accuracy_scores =[]
        for seed in range(1):
            np.random.seed(seed)
            random.seed(seed)
            tree_model = DecisionTreeClassifier(max_depth=depth, criterion="gini", random_state=seed)
            cv = StratifiedKFold(n_splits=3, random_state=seed, shuffle=True)
            cv_scores = cross_val_score(tree_model, X, y, cv=cv, scoring=scoring)
            this_scores_mean.append(cv_scores.mean())
            this_scores_std.append(cv_scores.std())
            this_accuracy_scores.append(tree_model.fit(X, y).score(X, y))
        
        cv_scores_mean.append(np.mean(np.array(this_scores_mean)))
        cv_scores_std.append(np.sqrt(np.mean(np.array([i**2 for i in this_scores_std]))))
        accuracy_scores.append(np.mean(np.array(this_accuracy_scores)))
        del this_scores_mean
        del this_scores_std
        del this_accuracy_scores
    cv_scores_mean = np.array(cv_scores_mean)
    cv_scores_std = np.array(cv_scores_std)
    accuracy_scores = np.array(accuracy_scores)
    return cv_scores_mean, cv_scores_std, accuracy_scores

def run_on_trees(X, y, X_test, y_test, scoring='balanced_accuracy'):
    param_dict = {
	"criterion":['gini'],
	"max_depth":[10],
	"splitter":['best', 'random'],
	"random_state":range(1000)
    }
    grid = sklearn.model_selection.GridSearchCV(DecisionTreeClassifier(), param_grid = param_dict, verbose = 1, refit = True, n_jobs = -1)
    grid.fit(X, y)
    print(grid.cv_results_)
    print(grid.best_estimator_)
    print(np.mean(this_scores_mean), np.std(this_scores_mean))
  
# function for plotting cross-validation results
def plot_cross_validation_on_trees(depths, cv_scores_mean, cv_scores_std, accuracy_scores, title):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    ax.plot(depths, cv_scores_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(depths, cv_scores_mean-2*cv_scores_std, cv_scores_mean+2*cv_scores_std, alpha=0.2)
    ylim = plt.ylim()
    ax.plot(depths, accuracy_scores, '-*', label='train accuracy', alpha=0.9)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(depths)
    ax.legend()
    plt.show()


def run_single_tree(X_train, y_train, X_test, y_test, depth, id):
    model = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)
    predict_train = model.predict(X_train)
    accuracy_train = sklearn.metrics.balanced_accuracy_score(y_train, predict_train)
    predict_test = model.predict(X_test)
    print(y_test)
    print(predict_test)
    #accuracy_test = sklearn.metrics.balanced_accuracy_score(y_test, predict_test)#BA_compute(y_test, predict_test, id)
    accuracy_test = BA_compute(y_test, predict_test, id)
    print('Single tree depth: ', depth)
    print('Accuracy, Training Set: ', round(accuracy_train*100,5), '%')
    print('Accuracy, Test Set: ', round(accuracy_test*100,5), '%')
    return accuracy_train, accuracy_test, model

def augment_func(X_train_old, y_train_old, size, noise):
    X_train_std = np.std(X_train_old, axis=0)
    X_train = np.zeros((X_train_old.shape[0] * size, X_train_old.shape[1]))
    y_train = np.zeros((y_train_old.shape[0] * size, 3))

    idx = X_train_old.shape[0]
    X_train[:idx,:] = X_train_old
    y_train[:idx,:] = y_train_old
    for idx in range(X_train_old.shape[0], X_train.shape[0]):
        sample_idx = random.randint(0, X_train_old.shape[0]-1)
        X_train[idx,:] = X_train_old[sample_idx, :] + np.random.randn((X_train_old.shape[1])) * X_train_std
        y_train[idx, :] = y_train_old[sample_idx,:]
    return X_train, y_train

################FLAGS####################
plot_inputdistr = False
plot_targetdistr = False
PCA_flag = False
tsne_flag = False
logistic = False
normalize=True
decision_tree = True
augment = True
low_populated_classes_remove = False

train_dataloader_input = WoNoWa_DataLoader('../data/', [2,3,4, 5, 6, 8, 9, 10, 13, 15, 17, 11], [3], True, False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataloader_input,
                                                       batch_size=10000,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=0)

test_dataloader_input = WoNoWa_DataLoader('../data/', [7, 16, 14], [3], True, False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataloader_input,
                                                       batch_size=1000,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=0)

clf = tree.DecisionTreeClassifier()
in_train = 0
target_train = 0
in_test = 0
target_test = 0

target_id = {0: "specialization", 1:"credibility", 2: "coordination"}

###added

for batch_idx, (in_tensor, target, audio_feat, spacial_feat) in enumerate(train_loader):
    X_train = torch.cat([audio_feat,spacial_feat], dim=1)
    y_train = target#specialization, credibility, coordination
    break;
for batch_idx, (in_tensor, target, audio_feat, spacial_feat) in enumerate(test_loader):
    X_test = torch.cat([audio_feat,spacial_feat], dim=1)
    y_test = target#specialization, credibility, coordination
    break;

X_train = X_train.numpy()
y_train = y_train.numpy()

if plot_inputdistr:
    X_train_mean = torch.mean(X_train, dim=0)
    X_train_std = torch.std(X_train, dim=0)
    labels = ["distancies3", "performance3", "num_Occupe.zone.commune3", "time_perc_Occupe.zone.commune3", "mean_time_Occupe.zone.commune3",
                "num_Entre.dans.la.zone.d.autrui3", "time_perc_Entre.dans.la.zone.d.autrui3", "mean_time_Entre.dans.la.zone.d.autrui3", "num_Occupe.sa.zone3",
                "time_perc_Occupe.sa.zone3", "mean_time_Occupe.sa.zone3", "n.formations_autre3", "time_perc_autre3", "mean_time_autre3", "n.formations_Triangulaire3",
                "time_perc_Triangulaire3", "mean_time_Triangulaire3", "n.formations_Semi.circulaire3", "time_perc_Semi.circulaire3", "mean_time_Semi.circulaire3",
                "qom3.mean", "V_mean3", "V_entropyMean3", "QoM_mean3", "QoM_entropyMean3", "directness_mean3",
                "TST.min3", "TSL.min3", "ASL3", "TAI.min3", "TSI.min3", "TSI.TAI3"]
    print(len(labels))
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(labels)), X_train_mean, yerr=X_train_std, alpha=0.5, align='center', ecolor='black', capsize=10)
    ax.set_ylabel('Value')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(len(labels)))
    ax.xaxis.set_tick_params(labelsize=6)
    ax.set_xticklabels(labels, rotation=90)
    #ax.set_title('Extracted features from WoNoWa')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    #plt.savefig('wonowa_distr.png')
    plt.savefig('wonowa_distr.png', format='png', bbox_inches = 'tight', dpi=1200)
    plt.show()
    #plt.cla()
    #error()

if plot_targetdistr:
    distr = torch.zeros (5, 3)
    for j in range(y_train.shape[0]):
        distr[y_train[j, 0], 0]+=1
        distr[y_train[j, 1], 1]+=1
        distr[y_train[j, 2], 2]+=1
    labels = ["score 1", "score 2", "score 3", "score 4", "score 5"]
    print(len(labels))
    # Build the plot
    fig, ax = plt.subplots()
    rects1 =ax.bar(np.arange(len(labels), dtype=float)-0.3, distr[:,0], width = 0.3, alpha=0.5, align='center', ecolor='black', capsize=10, label='specialization')
    rects2 =ax.bar(np.arange(len(labels)), distr[:,1], width = 0.3, alpha=0.5, align='center', ecolor='black', capsize=10, label='credibility')
    rects3 =ax.bar(np.arange(len(labels), dtype=float)+0.3, distr[:,2], width = 0.3, alpha=0.5, align='center', ecolor='black', capsize=10, label='coordination')
    #ax.set_ylabel('Value')
    #ax.set_yscale('log')
    ax.set_xticks(np.arange(len(labels)))
    ax.xaxis.set_tick_params(labelsize=6)
    ax.set_xticklabels(labels, rotation=90)

    ax.bar_label(rects1, padding=0)
    ax.bar_label(rects2, padding=0)
    ax.bar_label(rects3, padding=0)
    #ax.set_title('Extracted features from WoNoWa')
    ax.yaxis.grid(True)
    plt.legend()
    # Save the figure and show
    #plt.tight_layout()
    #plt.savefig('wonowa_distr.png')
    plt.savefig('wonowa_target_train.png', format='png', bbox_inches = 'tight', dpi=1200)
    plt.show()
    #plt.cla()
    #error()

if normalize:
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

if PCA_flag:
    total_dimensions = X_train.shape[1]

    pca = PCA(n_components = total_dimensions)
    pca.fit(X_train)
    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
    cum_var_explained = np.cumsum(percentage_var_explained)
    plt.plot(range(1, len(cum_var_explained[:])+1), cum_var_explained[:], '--', linewidth=2, label = 'without normalization')

    pca = PCA(n_components = total_dimensions)

    pca.fit(X_train)
    percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
    cum_var_explained = np.cumsum(percentage_var_explained)
    plt.plot(np.arange(1, len(cum_var_explained[:])+1), cum_var_explained[:], linewidth=2, label = 'with normalization')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative variance')
    plt.axhline(y=0.95, linestyle=':', color='red')
    plt.text(25, 0.92, '95% cut-off threshold', color = 'red', fontsize=8)
    plt.legend()
    plt.savefig('PCA.png', format='png', bbox_inches = 'tight')
    #plt.cla()
    i = 0
    while cum_var_explained[i] < 0.95:
        i += 1
    print('Components:', i)
    pca = PCA(n_components=i)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

if low_populated_classes_remove:
    filtered = np.where(np.prod((y_train >= 2).numpy(), axis = 1) == 1)[0]
    X_train = X_train[filtered, :]
    y_train = y_train[filtered, :]

    filtered = np.where(np.prod((y_test >= 2).numpy(), axis = 1) == 1)[0]
    X_test = X_test[filtered, :]
    y_test = y_test[filtered, :]

if augment:
    np.random.seed(42)
    random.seed(42)
    size = 1000
    noise = 0.01
    X_train, y_train = augment_func(X_train, y_train, size, noise)

for this_id in target_id:
    #score_labels = ["score 2", "score 3", "score 4", "score 5"]
    #if this_id == 2:
    score_labels = ["score 1", "score 2", "score 3", "score 4", "score 5"]
    if tsne_flag:
        from sklearn.manifold import TSNE
        X_2d = TSNE(n_components=2, random_state=0).fit_transform(X_train)
        plt.figure(figsize=(6, 5))
        colors = 'r', 'g', 'b', 'c', 'm'#, 'y', 'k', 'w', 'orange', 'purple'
        for i, c, label in zip(range(len(score_labels)), colors, score_labels):
            plt.scatter(X_2d[y_train[:,this_id] == i, 0], X_2d[y_train[:,this_id] == i, 1], c=c, label=label)
        plt.legend()
        plt.title(target_id[this_id])
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.savefig('tsne_'+target_id[this_id]+'.png', bbox_inches = 'tight', format='png')
        plt.close()


    if logistic:
        logreg = sklearn.linear_model.LogisticRegression(max_iter=1000)
        logreg.fit(X_train, y_train[:,this_id])
        #Ypred = logreg.predict(pcad_data_test)
        acc_log = logreg.score(X_train,y_train[:,this_id])
        print("TRAIN:", acc_log)
        expected = logreg.predict(X_test)
        this_BA = BA_compute(expected, y_test, this_id)
        #acc_log = logreg.score(X_test,y_test[:,this_id])
        print("TEST:", this_BA)

    if decision_tree:
        # fitting trees of depth 1 to 24
        run_on_trees(X_train, y_train[:,this_id], X_test, y_test[:,this_id], scoring='balanced_accuracy')
        #sm_tree_depths = range(1,11)
        #sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores = run_cross_validation_on_trees(X_train, y_train[:,this_id], sm_tree_depths)

        # plotting accuracy
        #plot_cross_validation_on_trees(sm_tree_depths, sm_cv_scores_mean, sm_cv_scores_std, sm_accuracy_scores, 
        #                            'Accuracy per decision tree depth on training data')
        '''
        idx_max = sm_cv_scores_mean.argmax()
        sm_best_tree_depth = sm_tree_depths[idx_max]
        sm_best_tree_cv_score = sm_cv_scores_mean[idx_max]
        sm_best_tree_cv_score_std = sm_cv_scores_std[idx_max]
        print('The depth-{} tree achieves the best mean cross-validation accuracy {} +/- {}% on training dataset'.format(
            sm_best_tree_depth, round(sm_best_tree_cv_score*100,5), round(sm_best_tree_cv_score_std*100, 5)))

        sm_best_tree_accuracy_train, sm_best_tree_accuracy_test, model = run_single_tree(X_train, y_train[:,this_id], 
                                                                                X_test, y_test[:,this_id], 
                                                                                sm_best_tree_depth, this_id)
        if not PCA_flag:
            export_name = target_id[this_id]+"_"+str(sm_best_tree_accuracy_test)+"_tree"
            export_graphviz(
            model,
            out_file=(export_name + ".dot"),
            feature_names=["distancies3", "performance3", "num_Occupe.zone.commune3", "time_perc_Occupe.zone.commune3", "mean_time_Occupe.zone.commune3",
                "num_Entre.dans.la.zone.d.autrui3", "time_perc_Entre.dans.la.zone.d.autrui3", "mean_time_Entre.dans.la.zone.d.autrui3", "num_Occupe.sa.zone3",
                "time_perc_Occupe.sa.zone3", "mean_time_Occupe.sa.zone3", "n.formations_autre3", "time_perc_autre3", "mean_time_autre3", "n.formations_Triangulaire3",
                "time_perc_Triangulaire3", "mean_time_Triangulaire3", "n.formations_Semi.circulaire3", "time_perc_Semi.circulaire3", "mean_time_Semi.circulaire3",
                "qom3.mean", "V_mean3", "V_entropyMean3", "QoM_mean3", "QoM_entropyMean3", "directness_mean3",
                "TST.min3", "TSL.min3", "ASL3", "TAI.min3", "TSI.min3", "TSI.TAI3"],
            class_names=score_labels,
            filled=True,
            )
            os.system("dot -Tpng "+ export_name +".dot -o "+target_id[this_id]+".png")
        '''