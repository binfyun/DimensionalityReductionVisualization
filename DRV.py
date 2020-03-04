
from __future__ import print_function
import time
from datetime import datetime

import subprocess
import os
#os.environ["MODIN_ENGINE"] = "ray"
import sys
import math
import operator
import csv
import os.path
# from distributed import Client
# client = Client(n_workers=6)
#import modin.pandas as pd
import pandas as pd
import numpy as np
import itertools
from itertools import chain
from itertools import groupby
import matplotlib as mpl
mpl.use('Agg')
import plotly
import plotly.graph_objs as go
import plotly.express as px


#from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import umap
import hdbscan
#import umap.plot

#%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from adjustText import adjust_text

in_file = sys.argv[1]


## Dimentional reduction and visualization:
#### UMAP ####
def vizUMAP(value, dataset, suffix):

#### Quick visual with Enhanced clustering and labels using hdbscan ###############################################
    #standard_embedding = umap.UMAP(random_state=42).fit_transform(value)
    #clusterable_embedding = umap.UMAP(n_neighbors=30,min_dist=0.0,n_components=2,random_state=42).fit_transform(value)
    #labels = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=500).fit_predict(clusterable_embedding)
    #clustered = labels
    # plt.figure(figsize=(16,10))
    # plt.scatter(standard_embedding[~clustered, 0],
    #             standard_embedding[~clustered, 1],
    #             c=(0.5, 0.5, 0.5),
    #             s=0.1,
    #             alpha=0.5)
    # plt.scatter(standard_embedding[clustered, 0],
    #             standard_embedding[clustered, 1],
    #             c=labels[clustered],
    #             s=0.1,
    #             cmap='Spectral')
    # plt.tight_layout()
    # plt.savefig("UmapEnhanced_hdbscan_cl.png")
    # plt.close('all')
##################################################################################################

############ Kmean to cluster ###############################
    # clusters = KMeans(n_clusters=3)
    # clusters.fit(embedding)

###################### UMAP with hdbscan ################################################
    #embedding = umap.UMAP(n_neighbors=25, min_dist=0.3, random_state=42, metric='dice').fit_transform(value)
    embedding = umap.UMAP(n_components=2, random_state=42, metric='dice').fit_transform(value)
    labels = hdbscan.HDBSCAN(min_samples=60,min_cluster_size=15).fit_predict(embedding)
    clustered = labels
    #umap.plot.points(embedding)
    #umap.plot.diagnostic(embedding, diagnostic_type='pca')
######################################################################################

#############################################################
    ## Plots:
    dataset['umap-one'] = embedding[:,0]
    dataset['umap-two'] = embedding[:,1] 
    #dataset['pca-three'] = pca_result[:,2]

### Append pesudogene and class labels ###################################
    dataset_gene = pd.read_csv("byGene_fc_AF_mod.txt", sep="\t", header=0, index_col=0)
    dataset_race = pd.read_csv("Race_information_on_25K_patients.txt", sep="\t", header=0, index_col=0)
    dataset.merge(dataset_race, left_on='ID', right_on='MRN', left_index=True)
    # with open("byGene_fc_AF_mod.txt", "r") as pseudo:
    #     for items in pseudo.readlines():
    #         item = items.strip().split("\t")
    #         for index_label, row_series in dataset.iterrows():
    #             if item[0] == index_label.strip('-N'):
    #                 dataset.at[index_label , 'Gene'] = item[1]
    # with open("Race_information_on_25K_patients.txt", "r") as race:
    #     for items in race.readlines():
    #         item = items.strip().split("\t")
    #         for index_label, row_series in dataset.iterrows():
    #             if item[0] == index_label.strip('-N'):
    #                 dataset.at[index_label , 'Race'] = item[1]
##########################################################################

############# Plot UMAP with gene annotation ########################
    plt.figure(figsize=(16,10))
    ax = sns.scatterplot(
        x="umap-one", 
        y="umap-two",
        #hue=["c{}".format(c) for c in clusters.labels_], #### Kmeans
        #hue=["c{}".format(c) for c in clustered],  #### Hdbscan
        hue = "Race",
        #palette=sns.color_palette("hls", 10),
        data=dataset,
        legend="full",
        alpha=0.3
    )

    texts = []
    for line in range(0,dataset.shape[0]):
        if not "nan" in str(dataset['Gene'][line]):
            texts.append(ax.text(dataset['umap-one'][line], dataset['umap-two'][line], dataset['Gene'][line], horizontalalignment='left', fontsize=7, color='black', weight='semibold'))
    adjust_text(
        texts, force_points=0.2, force_text=0.2,
        expand_points=(1, 1), expand_text=(1, 1),
        arrowprops=dict(arrowstyle="-", color='black', lw=0.5)
        )

    #plt.tight_layout()
    figure = ax.get_figure()
    figure.savefig("umap_2pc_ran42_dice_%s_w_gene_label_s7_semibold_autotextadjust_hdbscan_msmple120_mcls15.png"%suffix)
    plt.close('all')
    pseudo.close()
    race.close()

############################## PCA ###########################################3
def vizPCA(value, dataset, suffix):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(value)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
################# Kmeans clutsering ####################################
    # clusters = KMeans(n_clusters=5)
    # clusters.fit(pca_result)
################# hdbscan clustering ####################################
    labels = hdbscan.HDBSCAN(min_samples=10,min_cluster_size=15).fit_predict(pca_result)
    print(labels)
    #clustered = (labels >= 0)  ## do show noise cluster
#########################################################################
    
    clustered = labels
    ## Plots:
    dataset['pca-one'] = pca_result[:,0]
    dataset['pca-two'] = pca_result[:,1] 
    #dataset['pca-three'] = pca_result[:,2]

########## Plot PCA with annotated data points #################################
###### One plot for pseudogene ################
    #colors = {'NA':'grey'}
    plt.figure(figsize=(16,10))
    bx = sns.scatterplot(
        x="pca-one", 
        y="pca-two",
        #hue=["c{}".format(c) for c in clusters.labels_], ############ Kmean
        hue=["c{}".format(c) for c in clustered],  ################### HDBSCAM 
        #palette=sns.color_palette("hls", 10),
        #palette=colors,
        data=dataset,
        legend="full",
        alpha=0.3
    )

    texts = []
    for line in range(0,dataset.shape[0]):
        texts.append(bx.text(dataset['pca-one'][line], dataset['pca-two'][line], dataset.index[line], horizontalalignment='left', fontsize=7, color='black', weight='semibold'))
    adjust_text(
        texts, force_points=0.2, force_text=0.2,
        expand_points=(1, 1), expand_text=(1, 1),
        arrowprops=dict(arrowstyle="-", color='black', lw=0.5)
        )

    # def label_point(x, y, val, ax):
    #     #texts = []
    #     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    #     for i, point in a.iterrows():
    #         if not "nan" in str(point['val']):
    #             #print(point)
    #             ax.text(point['x']+.02, point['y'], str(point['val']), fontsize=7, weight='semibold')

    # label_point(dataset['pca-one'], dataset['pca-two'], dataset['Gene'], plt.gca())

    #plt.tight_layout()
    figure = bx.get_figure()
    figure.savefig("pca_%s_clustering.png"%suffix)
    plt.close('all')

#### t-SNE ######################################################
def vizTSNE(value, dataset, suffix):
    time_start = time.time()
    startTime = datetime.now()
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=10000)
    tsne_results = tsne.fit_transform(value)
    clusters = KMeans(n_clusters=5)
    clusters.fit(tsne_results)
    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
    print(datetime.now() - startTime)

    dataset['tsne-pca50-one'] = tsne_results[:,0]
    dataset['tsne-pca50-two'] = tsne_results[:,1]
    #dataset['tsne-2d-one'] = tsne_results[:,0]
    #dataset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    bx = sns.scatterplot(
        x="tsne-pca50-one", 
        y="tsne-pca50-two",
        #x="tsne-2d-one", 
        #y="tsne-2d-two",
        hue=["c{}".format(c) for c in clusters.labels_],
        #palette=sns.color_palette("hls", 10),
        data=dataset,
        legend="full",
        alpha=0.3
    )
    #plt.tight_layout()
    figure = bx.get_figure()
    figure.savefig("t_sne_%s.png"%suffix)
    plt.close('all');

def VIZ(infile):
################# Encode data ######################################################

    ## OneHot encoding
    startTime = datetime.now()
    print("Starting feature encoding")
    del_pheno = pd.read_csv(infile, sep=",", header=0, index_col=0)
    del_pheno['Starting AA'] = 'S' + del_pheno['Starting AA'].astype(str)
    del_pheno['Ending AA'] = 'E' + del_pheno['Ending AA'].astype(str)
    print(del_pheno)
    varieties = list(del_pheno.index)
    del_pheno_encoded = pd.get_dummies(del_pheno)
    del_pheno_values = del_pheno_encoded.values
    print(del_pheno_values)
    print("Feature encoded!", datetime.now() - startTime)


#########################################################################################################

    ## Fit data to umap, PCA and, T-sne

    # startTime = datetime.now()
    # print("Starting PCA")
    # dummy_encoded_pca_df=vizPCA(del_pheno_values, del_pheno_encoded, "del_genotype_2cp_")
    # print("PCA complete!", datetime.now() - startTime)

    # back_from_dummies(dummy_encoded_pca_df)

    # startTime = datetime.now()
    # print("Starting UMAP")
    # vizUMAP(data_subset_genotype, FP_dfs_encoded, "2014-2019_genotype_nbgr30_dist03_dice_hdbscan")
    # print("UMAP complete!", datetime.now() - startTime)
    
    #vizTSNE(pca50_vaf, FP_dfs_vaf, "vaf_prp100_nter10000_pca50_2cp")
    # startTime = datetime.now()
    # print("Starting t-SNE with PCA50")
    # vizTSNE(pca50_genotype, FP_dfs_encoded, "2014-2019_genotype_prp50_nter10000_pca50_2cp_cl5")
    # print("t-SNE with PCA50 complete!", datetime.now() - startTime)
    
    #vizTSNE(data_subset_vaf, FP_dfs_vaf, "vaf_prp100_nter10000_2cp")
    # startTime = datetime.now()
    # print("Starting t-SNE")
    # vizTSNE(data_subset_genotype, FP_dfs_encoded, "2014-2019_genotype_prp50_nter10000_2cp")
    # print("t-SNE complete!", datetime.now() - startTime)

## Heirachy CL dendrogram
    mergings = linkage(del_pheno_values, method='complete')
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax = dendrogram(mergings,labels=varieties,leaf_rotation=90,leaf_font_size=7)
    plt.gcf()
    plt.savefig("Hierarchy_CL.png")
    plt.close('all')

def main():    
    VIZ(in_file)

if __name__ == '__main__':
    main()



