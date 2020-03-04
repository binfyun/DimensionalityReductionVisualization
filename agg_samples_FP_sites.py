
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


def transpose(file_list):
    dfs_vaf = []
    dfs_genotype = []
    for item in file_list:
        #if "-N" in item and not "PoolNormal" in item:
        fpfile = item.split("/")[8]
        #print(fpfile)
        ID  = fpfile.split("_")[0].strip().split("-")[0]
        #print(ID)
        candidate = pd.read_csv(item, sep="\t", header=None, index_col=0)
        id_features = candidate.T
        #print(id_features)
        id_features_genotype = id_features.drop([1,3])
        #id_features_vaf = id_features.drop([1,2])
        id_features_genotype.insert(loc=0, column='ID', value=ID)
        #id_features_vaf.insert(loc=0, column='ID', value=ID)
        id_features_genotype_final = id_features_genotype.set_index('ID')
        #id_features_vaf_final = id_features_vaf.set_index('ID')
        dfs_genotype.append(id_features_genotype_final)
        #dfs_vaf.append(id_features_vaf_final)
        FP_dfs_genotype = pd.concat(dfs_genotype)
        #print(FP_dfs_genotype)
        #FP_dfs_vaf = pd.concat(dfs_vaf)
    #return (FP_dfs_genotype, FP_dfs_vaf);
    return FP_dfs_genotype
    #return dfs_genotype

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
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(value)
    print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))
################# Kmeans clutsering ####################################
    # clusters = KMeans(n_clusters=5)
    # clusters.fit(pca_result)
################# hdbscan clustering ####################################
    labels = hdbscan.HDBSCAN(min_samples=60,min_cluster_size=15).fit_predict(pca_result)
    print(labels)
    #clustered = (labels >= 0)
    clustered = labels
    ## Plots:
    dataset['pca-one'] = pca_result[:,0]
    dataset['pca-two'] = pca_result[:,1] 
    #dataset['pca-three'] = pca_result[:,2]
    dataset_fillna = dataset.fillna("NA")
    dataset_NAonly_gene = dataset_fillna[dataset_fillna['Gene']=="NA"]
    dataset_NAonly_race = dataset_fillna[dataset_fillna['race']=="NA"]
### Append pesudogene and class labels ###################################
    # startTime = datetime.now()
    # print("Getting pseudogenes")
    # with open("byGene_fc_AF_mod.txt", "r") as pseudo:
    #     for items in pseudo.readlines():
    #         item = items.strip().split("\t")
    #         for index_label, row_series in dataset.iterrows():
    #             if item[0] == index_label.strip('-N'):
    #                 dataset.at[index_label , 'Gene'] = item[1]
    # print("Pseudogene atached!", datetime.now() - startTime)
    # print(dataset)
    # startTime = datetime.now()
    # print("Getting race")
    # with open("Race_information_on_25K_patients.txt", "r") as race:
    #     for items in race.readlines():
    #         item = items.strip().split("\t")
    #         for index_label, row_series in dataset.iterrows():
    #             if item[0] == index_label.strip('-N'):
    #                 dataset.at[index_label , 'Race'] = item[1]
    # print("Race atached!", datetime.now() - startTime)
    # print(dataset)
    # dataset_CL = dataset
    # dataset_CL['cluster'] = labels
    # dataset_CL.to_csv(r'PCA_out_Gene_Race.txt', index=True, sep='\t', header=True)
##########################################################################

########## Plot PCA with annotated data points #################################
###### One plot for pseudogene ################
    colors = {'NA':'grey'}
    plt.figure(figsize=(16,10))
    bx = sns.scatterplot(
        x="pca-one", 
        y="pca-two",
        #hue=["c{}".format(c) for c in clusters.labels_], ############ Kmean
        #hue=["c{}".format(c) for c in clustered],  ################### HDBSCAM
        hue = "Gene",
        #palette=sns.color_palette("hls", 10),
        palette=colors,
        data=dataset_NAonly_gene,
        legend=False,
        alpha=0.3
    )
    ax = sns.scatterplot(
        x="pca-one", 
        y="pca-two",
        #hue=["c{}".format(c) for c in clusters.labels_], ############ Kmean
        #hue=["c{}".format(c) for c in clustered],  ################### HDBSCAM
        hue = "Gene",
        palette=sns.color_palette("hls", 16),
        data=dataset,
        legend="full",
        alpha=0.7
    )
    # texts = []
    # for line in range(0,dataset.shape[0]):
    #     if not "nan" in str(dataset['Gene'][line]):
    #         texts.append(ax.text(dataset['pca-one'][line], dataset['pca-two'][line], dataset['Gene'][line], horizontalalignment='left', fontsize=7, color='black', weight='semibold'))
    # adjust_text(
    #     texts, force_points=0.2, force_text=0.2,
    #     expand_points=(1, 1), expand_text=(1, 1),
    #     arrowprops=dict(arrowstyle="-", color='black', lw=0.5)
    #     )

    # def label_point(x, y, val, ax):
    #     #texts = []
    #     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    #     for i, point in a.iterrows():
    #         if not "nan" in str(point['val']):
    #             #print(point)
    #             ax.text(point['x']+.02, point['y'], str(point['val']), fontsize=7, weight='semibold')

    # label_point(dataset['pca-one'], dataset['pca-two'], dataset['Gene'], plt.gca())

    #plt.tight_layout()
    figure = ax.get_figure()
    figure.savefig("pca_%s_GeneCls.png"%suffix)
    plt.close('all')
#######  one plot for Race #########################
    Genelist = ["GNA11", "SMAD4"]
    Genelist2 = ["GNA11", "SMAD4", "NA"]
    dataset_GNA11_SMAD4 = dataset_fillna[dataset_fillna['Gene'].isin(Genelist)]
    dataset_GNA11_SMAD4_NA= dataset_NAonly_race[dataset_NAonly_race['Gene'].isin(Genelist2)]
    print(dataset_GNA11_SMAD4_NA)

##### 2D #####################################################
    markers = {"GNA11": "X", "SMAD4": "D"}
    colors = {'NA':'grey'}
    plt.figure(figsize=(16,10))
    bx = sns.scatterplot(
        x="pca-one", 
        y="pca-two",
        #hue=["c{}".format(c) for c in clusters.labels_], ############ Kmean
        #hue=["c{}".format(c) for c in clustered],  ################### HDBSCAM
        hue = "race",
        #palette=sns.color_palette("hls", 10),
        palette=colors,
        data=dataset_NAonly_race,
        legend=False,
        alpha=0.3
    )
    ax = sns.scatterplot(
        x="pca-one", 
        y="pca-two",
        #hue=["c{}".format(c) for c in clusters.labels_], ############ Kmean
        #hue=["c{}".format(c) for c in clustered],  ################### HDBSCAM
        hue = "race",
        palette=sns.color_palette("hls", 5),
        data=dataset,
        legend="full",
        alpha=0.5
    )
    cx = sns.scatterplot(
        x="pca-one", 
        y="pca-two",
        #hue=dataset_GNA11_SMAD4["Gene"],
        style=dataset_GNA11_SMAD4["Gene"],
        #size=dataset_GNA11_SMAD4["Gene"],
        color="black",
        markers = markers,
        s=80,
        #palette=sns.color_palette("hls", 2),
        data=dataset_GNA11_SMAD4,
        legend="full",
        alpha=0.8
    )
    # texts = []
    # print(dataset)
    # for line in range(0,dataset.shape[0]):
    #     #print(line)
    #     try:
    #         if str(dataset['Gene'][line]) in Genelist:
    #             texts.append(ax.text(dataset['pca-one'][line], dataset['pca-two'][line], dataset['Gene'][line], fontsize=7, color='abbi-nina-1125
    #             ', weight='semibold')) #horizontalalignment='left'
    #     except KeyError:
    #         continue
    # #adjust_text(texts)
    # adjust_text(
    #     texts, force_points=0.2, force_text=0.2,
    #     expand_points=(1, 1), expand_text=(1, 1),
    #     arrowprops=dict(arrowstyle="-", color='black', lw=0.5),
    #     autoalign="y"
    #     )

    # def label_point(x, y, val, ax):
    #     #texts = []
    #     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    #     for i, point in a.iterrows():
    #         if not "nan" in str(point['val']):
    #             #print(point)
    #             ax.text(point['x']+.02, point['y'], str(point['val']), fontsize=7, weight='semibold')

    # label_point(dataset['pca-one'], dataset['pca-two'], dataset['Gene'], plt.gca())

    #plt.tight_layout()
    figure = ax.get_figure()
    figure.savefig("pca_%s_RaceCls_wGene_Anno_style_twoGenes.png"%suffix)
    plt.close('all')
###############################    3D         ####################################################
    #Set marker properties
    markercolor = dataset['race'].replace("ASIAN","cyan").replace("WHITE","red").replace("BLACK","green").replace("UNKNOWN","grey").replace("OTHER","lightblue")
    #markershape = dataset_GNA11_SMAD4['Gene'].replace("GNA11","square").replace("SMAD4","diamond")
    #Make Plotly figure
    fig1 = go.Scatter3d(x=dataset['pca-one'],
                        y=dataset['pca-two'],
                        z=dataset['race'],
                        text=dataset['Gene'],
                        marker=dict(color=markercolor,
                                    opacity=0.4,
                                    reversescale=False,
                                    colorscale="greys",
                                    size=3
                                    ),
                        line=dict(width=0.02),
                        mode='markers')

    #Make Plot.ly Layout
    mylayout = go.Layout(
        scene=dict(xaxis=dict(title="PCA_1"),yaxis=dict(title="PCA_2"),zaxis=dict(title="Ethnicity")),
        annotations=[
            dict(
            showarrow=False,
            x=dataset['pca-one'],
            y=dataset['pca-two'],
            text='GNA11',
            xanchor="left",
            xshift=10,
            opacity=0.7) 
            # dict(
            # x="2017-02-10",
            # y="B",
            # z=4,
            # text="Point 2",
            # textangle=0,
            # ax=0,
            # ay=-75,
            # font=dict(color="black",size=12),
            # arrowcolor="black",
            # arrowsize=3,
            # arrowwidth=1,
            # arrowhead=1), 
            # dict(
            # x="2017-03-20",
            # y="C",
            # z=5,
            # ax=50,
            # ay=0,
            # text="Point 3",
            # arrowhead=1,
            # xanchor="left",
            # yanchor="bottom")
            ])

    #Plot and save html
    plotly.offline.plot({"data": [fig1],
                        "layout": mylayout},
                        auto_open=False,
                        filename=("3DPlot_with_hoverGeneText.html"))
    # pseudo.close()
    # race.close()
    #return dataset_CL
############################################################
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


def parsing(roots):
    pools = []
    FP_files = []
    #for root in roots:
    for dir in os.listdir(roots):
        #print dir
        if dir.startswith(("IMPACTv6-CLIN", "IMPACTv5-CLIN", "IMPACTv4-CLIN", "IMPACTv3-CLIN")):
            pools.append(dir)
            in_dir = roots+str(dir)+"/"+"DEFAULT"+"/"
            #print(in_dir)
            try:
                for files in os.listdir(in_dir):
                    if files.endswith("BR.FP.summary.txt"):
                        if files.split("-")[0].isdigit() and "-N_bc" in files:
                            fp_file = in_dir+files
                            print(fp_file)
                            FP_files.append(fp_file)
                            #print(FP_files)
                    else:
                        continue
            except Exception:
                continue
            
    return FP_files

def back_from_dummies(df):
    result_series = {}

    # Find dummy columns and build pairs (category, category_value)
    dummmy_tuples = [(col.split("_")[0],col) for col in df.columns if "_" in col]

    # Find non-dummy columns that do not have a _
    non_dummy_cols = [col for col in df.columns if "_" not in col]

    # For each category column group use idxmax to find the value.
    for dummy, cols in groupby(dummmy_tuples, lambda item: item[0]):

        #Select columns for each category
        dummy_df = df[[col[1] for col in cols]]

        # Find max value among columns
        max_columns = dummy_df.idxmax(axis=1)

        # Remove category_ prefix
        result_series[dummy] = max_columns.apply(lambda item: item.split("_")[1])

    # Copy non-dummy columns over.
    for col in non_dummy_cols:
        result_series[col] = df[col]

    # Return dataframe of the resulting series
    return pd.DataFrame(result_series)

def get_all_data():
    years = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]
    #years = [sys.argv[1], sys.argv[2]]
    ## /dmp/dms/qc-data/IMPACT/2019/IMPACTv6-CLIN-20190597/DEFAULT ## Full path pattern
    ## IMPACTv6-CLIN-20190597_copynumber_segclusp.genes.txt, IMPACTv6-CLIN-20190597_copynumber_segclusp.nvn.genes.txt ## T and N cn files pattern
    roots = []
    for item in years:
        year = "/dmp/dms/qc-data/IMPACT/%s/"%item
        roots.append(year)
    print(roots)

    dfs_genotype = []
    for item in roots:
        startTime = datetime.now()
        print("Walking", item)
        FP_files=parsing(item)
        print('Directrory Walk done! Time elapsed:', datetime.now() - startTime)

        startTime = datetime.now()
        print("Starting DF transformation on", item)
        FP_dfs_genotype = transpose(FP_files)
        print("Dataframe conversion and transpose done!", datetime.now() - startTime)
        dfs_genotype.append(FP_dfs_genotype)
        FP_dfs = pd.concat(dfs_genotype)
    FP_dfs.to_pickle("FP_genotype.pkl")
    #FP_dfs_vaf.to_pickle("FP_vaf.pkl")

def VIZ(df):
################# Read in Pickle and Encode data ######################################################
    startTime = datetime.now()
    print("Reading in pickle")
    FP_dfs_genotype = pd.read_pickle(df)
    #FP_dfs_vaf = pd.read_pickle("FP_vaf.pkl")
    print("Pickle readin Done!", datetime.now() - startTime)
    #data_subset_vaf = FP_dfs_vaf.values
    varieties = list(FP_dfs_genotype.index)
    ## OneHot encoding
    startTime = datetime.now()
    print("Starting feature encoding")
    del_pheno = pd.read_csv("Race_information_on_25K_patients.txt", sep="\t", header=0, index_col=0)
    FP_dfs_encoded = pd.get_dummies(FP_dfs_genotype)
    #print(FP_dfs_encoded)
    print("Feature encoded!", datetime.now() - startTime)
    #data_subset_genotype = FP_dfs_encoded.values
    #print(data_subset_genotype)
    FP_dfs_encoded.reset_index(level=0, inplace=True)
    dataset_gene = pd.read_csv("byGene_fc_AF_mod.txt", sep="\t", header=0, index_col=False)
    dataset_race = pd.read_csv("Race_information_on_25K_patients.txt", sep="\t", header=0, index_col=False)

    dataset_race['MRN'] = dataset_race['MRN'].astype(str)
    dataset_gene['MRN'] = dataset_gene['MRN'].astype(str)
    FP_dfs_encoded['ID'] = FP_dfs_encoded['ID'].astype(str)
    FP_dfs_encoded_clean = FP_dfs_encoded.drop_duplicates(subset=['ID'])
    #FP_dfs_encoded_clean = FP_dfs_encoded.drop_duplicates(keep='last')
    FP_dfs_encoded_clean_indexID = FP_dfs_encoded_clean.set_index('ID')
    data_subset_genotype = FP_dfs_encoded_clean_indexID.values    
    #print(data_subset_genotype)
    #FP_dfs_encoded_clean.to_csv(r'PCA_original.txt', index=False, sep='\t', header=True)
    dataset_anno_race = FP_dfs_encoded_clean.merge(dataset_race, how='left', left_on='ID', right_on='MRN')
    #duplicateRowsDF = dataset_anno_race[dataset_anno_race.duplicated(['ID'])]
    #print(duplicateRowsDF)
    #print(dataset_anno_race)
    dataset_anno_race_gene = dataset_anno_race.merge(dataset_gene, how='left', left_on='ID', right_on='MRN') #, validate="1:m")
    dataset_anno_race_gene_clean = dataset_anno_race_gene.drop_duplicates(subset=['ID'])
    dataset_anno_race_gene_final = dataset_anno_race_gene_clean.drop(['MRN_x', 'MRN_y', 'Tumor', 'N_fc', 'AF'], axis=1)
    #duplicateRowsDF = dataset_anno_race_gene_final[dataset_anno_race_gene_final.duplicated(['ID'])]
    #print(duplicateRowsDF)
    #print(dataset_anno_race_gene_final)
    #dataset_anno_race_gene_final.to_csv(r'PCA_race_gene.txt', index=False, sep='\t', header=True)   ########## Write out the final DF

    #dataset_race.index = dataset_race.index.map(str)
    #dataset_anno_race_1 = FP_dfs_encoded.merge(dataset_race, how='left', left_index=True, right_index=True)
    #print(dataset_anno_race_1)
#########################################################################################################

    ## Fit data to umap, PCA and, T-sne
    #pca50_vaf=vizPCA(data_subset_vaf, FP_dfs_vaf, "vaf_2cp")
    startTime = datetime.now()
    print("Starting PCA and PCA50")
    dummy_encoded_pca_df=vizPCA(data_subset_genotype, dataset_anno_race_gene_final, "2014-2019_genotype_2cp_")
    print("PCA complete!", datetime.now() - startTime)

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
    # mergings = linkage(data_subset_genotype, method='complete')
    # fig, ax = plt.subplots(1, 1, figsize=(100,100))
    # ax = dendrogram(mergings,labels=varieties,leaf_rotation=90,leaf_font_size=6)
    # plt.gcf()
    # plt.savefig("Hierarchy_CL.png")
    # plt.close('all')

## Quick TSNEVIZ
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # from yellowbrick.text import TSNEVisualizer
    # from yellowbrick.datasets import load_hobbies

    # startTime = datetime.now()
    # print("Starting QuickVIZ t-SNE")
    # tfidf = TfidfVectorizer()
    # X = tfidf.fit_transform(FP_dfs_genotype)

    # clusters = KMeans(n_clusters=3)
    # clusters.fit(X)
    # # Create the visualizer and draw the vectors
    # plt.figure(figsize=(20, 20))
    # tsne = TSNEVisualizer()
    # tsne.fit(X, ["c{}".format(c) for c in clusters.labels_])
    # plt.gcf()
    # plt.savefig("tSNEVIZ.png", dpi=600)
    # plt.close('all')
    # print("QuickVIZ t-SNE complete!", datetime.now() - startTime)

def main():
###### Parsing IMPACT cohorts #########################################
    #get_all_data()

    # startTime = datetime.now()
    # print("Reading in pickle")
    # FP_dfs_genotype = pd.read_pickle("FP_genotype.pkl")
    # print("Pickle readin Done!", datetime.now() - startTime)
    
    VIZ("FP_genotype.pkl")
###### Read in dataset and plot data points #######################################################
    # dataset = pd.read_csv("PCA_out.txt", sep="\t", header=0, index_col=0)
    # plt.figure(figsize=(16,10))
    # ax = sns.scatterplot(
    #     x="pca-one", 
    #     y="pca-two",
    #     palette=sns.color_palette("hls", 10),
    #     data=dataset,
    #     legend="full",
    #     alpha=0.3
    # )
######### Adding data point labels to the PCA plot #########################################
    # def label_point(x, y, val, ax):
    #     #texts = []
    #     a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    #     for i, point in a.iterrows():
    #         if not "nan" in str(point['val']):
    #             #print(point)
    #             ax.text(point['x']+.02, point['y'], str(point['val']), fontsize=7, weight='semibold')

    # label_point(dataset['pca-one'], dataset['pca-two'], dataset['Gene'], plt.gca())

######## Using Auto text adjust package #######################################################
    # texts = []
    # for line in range(0,dataset.shape[0]):
    #     if not "nan" in str(dataset['Gene'][line]):
    #         texts.append(ax.text(dataset['pca-one'][line]+0.2, dataset['pca-two'][line], dataset['Gene'][line], horizontalalignment='left', fontsize=6, color='black', weight='semibold'))
    # adjust_text(
    #     texts, force_points=0.2, force_text=0.2,
    #     expand_points=(1, 1), expand_text=(1, 1),
    #     arrowprops=dict(arrowstyle="-", color='black', lw=0.5)
    #     )

####### Arrow to the data points ##########################################################
    # for label, x, y in zip(dataset['Gene'], dataset['pca-one'], dataset['pca-two']):
    #     if not "nan" in str(label):
    #         plt.annotate(
    #             label,
    #             xy=(x, y), xytext=(-20, 20),
    #             textcoords='offset points', ha='right', va='bottom',
    #             #bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
    #             arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
##############################################################################################

    # # plt.tight_layout()
    # figure = ax.get_figure()
    # figure.savefig("pca_pseudogene_v3.png")
    # plt.close('all')
########################################################################################################

if __name__ == '__main__':
    main()



