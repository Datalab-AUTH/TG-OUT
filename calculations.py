import networkx as nx
import toolset
from tqdm import tqdm
from numpy import linalg
import numpy as np
from scipy import stats
import pickle
import pandas as pd
import time

'''RETURNS A LIST OF ALL THE DEGREES FOR EVERY NODE'''
def get_degree_frequency_list(G):
    deg_freq = []
    for n in G.nodes():
        if n != None:
            degree = nx.degree(G, n)
            deg_freq.append(degree)
    return deg_freq

def get_degree_frequency_list_for_bot_score(G,col,maxscore,minscore):
    deg_freq = []
    df = toolset.load_node_data_in_pandas(col.name,G)
    mask = df[(df['botscore']>minscore) & (df['botscore']<=maxscore)]
    allNodes = mask['label'].tolist()
    for n in G.nodes():
        if n != None and n in allNodes:
            degree = nx.degree(G, n)
            deg_freq.append(degree)
    return deg_freq

'''RETURNS A LIST OF ALL THE DEGREES FOR EVERY ATTRIBUTE THAT IS A LIST, E.G. HASHTAGS - URLS'''
def get_attribute_degree_frequency_list(G,attribute):
    att_deg_freq=[]
    for (node,attr) in G.nodes(data=True):
        if attribute in attr:
            degree=len(attr[attribute])
            att_deg_freq.append(degree)
    return att_deg_freq

'''RETURNS A LIST OF ALL THE DEGREES FOR EVERY ATTRIBUTE THAT IS A LIST, E.G. HASHTAGS - URLS WITH BOT SCORE!!'''
def get_attribute_degree_frequency_list_for_bot_score(G,col,attribute,maxscore,minscore):
    att_deg_freq=[]
    df = toolset.load_node_data_in_pandas(col.name, G)
    mask = df.loc[(df['botscore'] > minscore) & (df['botscore'] <= maxscore)]
    allNodes = mask['label'].tolist()
    # print ('mask :',mask.shape)
    for (node,attr) in G.nodes(data=True):
        if node in allNodes:
            if attribute in attr:
                degree=len(attr[attribute])
                att_deg_freq.append(degree)
    return att_deg_freq

'''RETURNS A LIST OF ALL DEGREES FOR EVERY ATTRIBUTE THAT IS A NUMBER,EG TWEETS,FRIENDS,FOLLOWERS'''
def get_unique_attribute_degree_frequency_list(G,col,attribute,maxscore,minscore):
    att_deg_freq = []
    df = toolset.load_node_data_in_pandas(col.name, G)
    mask = df.loc[(df['botscore'] > minscore) & (df['botscore'] <= maxscore)]
    allNodes = mask['label'].tolist()
    for (node, attr) in G.nodes(data=True):
        if node in allNodes:
            try:
                x = attr[attribute]
                att_deg_freq.append(x)
            except:
                print ('no info')
    return att_deg_freq

'''WRITES NODE INFORMATION IN TXT FILES - TWO TYPES: ONE FOR CUMULATIVE AND ONE FOR DELTA GRAPHS'''
def write_node_ico_stats(col,type='cumulative'):
    Glist = toolset.load_all_graphs_ico(col,type)
    if type=='cumulative':
        for G in Glist:
            file = open('stats/'+col.name+'/new/'+G.name + "_node_stats.txt", 'w')
            for node,attr in G.nodes(data=True):
                file.write( str(len(attr['hashtags'])) + ',' + str(len(attr['urls'])) + ',' +
                            str(attr['tweets']) + ',' +str(attr['retweets']) + ','+ str(attr['friends'])
                            + ',' + str(attr['followers']) + ',' + str(attr['degree']) + ','
                            +str(attr['weight']) + ',' +str(attr['label']) + "\n")
            file.close()
    elif type=='delta':
        for G in Glist:
            file = open('stats/'+col.name+'/'+  G.name + "_node_stats.txt", 'w')
            for node,attr in G.nodes(data=True):
                file.write( str(len(attr['hashtags'])) + ',' + str(len(attr['urls'])) + ',' +
                            str(attr['tweets']) + ',' + str(attr['degree']) + ','
                            +str(attr['weight']) + ',' + str(attr['label']) + "\n")
            file.close()
    else:
        pass

'''FUNCTION THAT WRITES ATTRIBUTE INDUCED SUBRAPHS -AIS- INFO IN TXT'''
def get_ais_stats(Glist,attribute,smax=True):
    import matrices_creation
    import fraudar as fr
    print ('...ais stats for:',attribute)
    Glist= Glist[1:]
    dataset = 'ico'
    infodict=pickle.load(open('FILE WITH ALL ATTRIBUTES, EG ALL HASHTAGS OR ALL URLS','rb'))
    botdict = infodict
    for G in Glist:
        text = open(dataset +'_'+ G.name + '_' + attribute + "_ais_stats.txt", 'w',encoding='utf-8')
        allAttributes=[]
        for n,attr in G.nodes(data=True):
            if attribute in attr:
                allAttributes.extend(attr[attribute])
        for a in set(allAttributes):
            try:
                subGraphList = [x for x, y in G.nodes(data=True) if a in y[attribute]]
                subgraph = G.subgraph(subGraphList)
                try:
                    coupled = matrices_creation.matrix_calculation(subgraph)
                    coupled = matrices_creation.coupled_matrix_using_weights(subgraph,attribute)
                    # tf = nt.produce_idf_dict(h)[1]
                    indicesList = [n for n in range(coupled.shape[0])]
                    fraudarScore = (fr.get_fraudar_score(coupled, indicesList))
                except (UnicodeEncodeError,MemoryError):
                    fraudarScore = 0.0
                    print ('fraudar error',a)
                volume = len(subgraph.nodes())
                mass = len(subgraph.edges())
                if volume==2 and mass>volume:
                    print (a,volume,mass)
                if mass>0:
                    if isinstance(nx.triangles(subgraph),dict)>0:
                        triangles = sum(nx.triangles(subgraph).values()) / 3
                    else:
                        triangles= nx.triangles(subgraph)
                else:
                    triangles=0
                componentList = []
                for c in nx.connected_components(subgraph):
                    componentList.append(len(c))
                isolated = len(componentList)
                Gcc = sorted(nx.connected_components(subgraph), key=len, reverse=True)
                try:
                    G0 = len(Gcc[0])
                except IndexError:
                    G0 = 0
                if subgraph.number_of_nodes() > 0 or subgraph.number_of_edges() > 0:
                    try:
                        A = nx.adjacency_matrix(subgraph)
                        B = A.todense()
                        U, s, V = linalg.svd(B)
                        sMax = np.amax(s)
                    except:
                        sMax = 0
                        print ('smax error',a)
                else:
                    sMax = 0
                text.write(
                    str(volume) + "," + str(mass) + "," + str(triangles) + "," + str(G0) + "," + str(sMax) + "," +
                    str(infodict[a])+","+str(botdict[a])+","+str(fraudarScore)+","+str(a) + "\n")
            except UnicodeEncodeError:
                print('Unicode Error')
        text.close()

