from create_graphs import get_date_of_oldest_newest_document
import datetime
import networkx as nx
from datetime import timedelta,datetime
import datetime
import pickle
import pandas as pd
import numpy as np
from community import community_louvain
from tqdm import tqdm
import sys
from sklearn import preprocessing
import toolset

def load_all_graphs_ico(col,type='cumulative'):
    PER_DAYS = 5
    Glist=[]
    bot_score = pickle.load(open('bot_score', 'rb'))
    start,end = get_date_of_oldest_newest_document(col)
    until = start + timedelta(days=PER_DAYS)
    while until.date()<= end.date():
        G = pickle.load(open('graphs/new/'+type+'_' +until.strftime('%m_%d_%Y'), 'rb'))
        subG = nx.subgraph(G,bot_score.keys())
        subG.name = type+'_'+ until.strftime('%m_%d_%Y')
        df = toolset.load_node_data_in_pandas('final_ico', subG)
        df1 = (df[df['tweets'] == 0])
        users = (df1['label'].tolist())
        suBG = nx.Graph(subG)
        suBG.remove_nodes_from(users)
        suBG.name = subG.name
        Glist.append(suBG)
        until = until + timedelta(days=PER_DAYS)
    return Glist

'''RETURNS A PANDAS DATAFRAME WITH VALUES OF THE TEXT FILES'''
def load_node_data_in_pandas(col,G):
    df=pd.DataFrame()
    col=='final_ico':
    botinfo = pickle.load(open('bot_score', 'rb'))
    df = pd.read_csv('stats/final_ico/new/' + G.name + '_node_stats.txt', sep=",", header=None,
                     names=['hashtags', 'urls', 'tweets','retweets', 'friends', 'followers',
                            'degree', 'weight', 'label'],
                     dtype={"label": str}, encoding="ISO-8859-1")
    df['botscore'] = df['label'].map(botinfo)
    return df

def load_node_ico_data_in_pandas(type,Gname):
    df=pd.DataFrame()
    if type=='cumulative':
        df=pd.read_csv('stats/final_ico/'+Gname+'_node_stats.txt', sep=",", header=None,
                     names=['hashtags', 'urls', 'tweets', 'friends', 'followers',
                            'degree', 'weight', 'label'],
                     dtype={"label": str}, encoding="ISO-8859-1")
    elif type=='delta':
        df = pd.read_csv('stats/'+Gname+'_node_stats.txt', sep=",", header=None,
                         names=['hashtags', 'urls', 'tweets',
                                'degree', 'weight', 'label'],
                         dtype={"label": str}, encoding="ISO-8859-1")
    return df

'''RETURNS A HEATMAP TXT FILE - TO BE USED WITH GGPLOT SCRIPT. USE ais==True IF YOU WANT TO GET HEATMAP FOR AIS GRAPH STATISTICS'''
def get_heatmap_file(x,y,type,Gname,ais):
    if ais==True:
        data = load_ais_data_in_pandas(type,Gname)
    else:
        data = load_node_data_in_pandas(type,Gname)
        data[x]+=1
        data[y]+=1
    newdf = data.groupby([x, y]).size().reset_index(name="Time")
    newdf.to_csv('heatplots/'+ x + "_" + y + '_'+Gname+'.txt', header=False, index=False)

'''RETURNS A PANDAS DATAFRAME WITH VALUES OF THE TEXT FILES FOR THE AIS STATISTICS'''
def load_ais_data_in_pandas(attribute,dataset,G,botmin,botmax):
    if dataset == 'final_ico':
        dataset='ico'
    else:
        dataset='elections'
    print ('...loading data...')
    names=['volume', 'mass', 'triangles', 'gcc', 'smax', 'party','botscore','fraudar', 'label']
    df = pd.read_csv('stats/paper/' +dataset+'_'+G.name+'_'+attribute+'_ais_stats.txt', sep=",",
                     header=None,names=names,dtype={"label": str,"party":str}, encoding="utf-8")
    df['density']=(2*df['mass'])/(df['volume']*(df['volume']-1))
    df['fraudarNorm'] = (df['fraudar'] - df['fraudar'].min()) / (df['fraudar'].max() - df['fraudar'].min())
    # print (df['density'].head(10))
    # print (df.head(10))
    df['smax']=df['smax'].apply(lambda x:round(x,5))
    df = df.dropna()
    df=df[df['volume']>10]
    # df = df[df['triangles'] > 0]
    df=df[df['density']>0]
    df=df[df['smax']>0]
    mask = df.loc[(df['botscore'] >= botmin) & (df['botscore'] <= botmax)]
    # print(df.head(10))
    return mask

'''FUNCTION TO PERFORM LOG BINNING - TAKEN BY SOAR'''
def log_fit_line(X,Y, nbins=20):
    import scipy
    """ Fits line by logarithmic binning of x values
        Args:
            x (array-list) : 1d array of positive values
            y (array-list) : 1d array of positive values
        Returns:
            (dict) : dictionary of bin_avg, line's x and y, slope, corr values
        """
    import math
    x= np.array(X.tolist())
    y = np.array(Y.tolist())
    try:
        assert np.min(x) != np.max(x)
    except:
        'x is a constant, binning failed in log_fit_line()'
        sys.exit(1)
    x_edges = np.logspace(
        base=10, start=np.log10(np.min(x)), stop=np.log10(np.max(x)), num=nbins + 1
    )
    x_bins, y_bins = np.zeros(nbins), np.zeros(nbins)
    for i in range(nbins):
        x_bins[i] = np.average(x_edges[i:i + 2])
        y_bins[i] = np.average(y[np.logical_and(x >= x_edges[i], x <= x_edges[i + 1])])
    select_bins = y_bins > 0
    select_bins[0] = False  # ignore the first bin as it has zero counts
    line = np.polyfit(np.log10(x_bins[select_bins]), np.log10(y_bins[select_bins]), 1)
    return {
        'm': line[0],
        'x': x_bins[select_bins],
        'bin_avg': y_bins[select_bins],
        'y': np.exp(np.poly1d(line)(np.log10(x_bins[select_bins]))),
        'corr': scipy.stats.pearsonr(
            np.log10(x_bins[select_bins]), np.log10(y_bins[select_bins])
        )[0]
    }
'''FUNCTION THAT RETURNS A COMMUNITY DICTIONARY FOR ALL NODES - BASED ON LOUVAIN ALGORITHM'''
def get_com_dict(graph):
    part = community_louvain.best_partition(graph)
    pickle.dump(part,open('louvain_community_bipartite_hashtags_'+graph.name,'wb'))
