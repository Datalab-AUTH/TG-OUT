import create_graphs as cg
import calculations as cal
import visualize_plots as vis
import clusterings
import pickle
import toolset
import numpy as np
from tqdm import tqdm
import networkx as nx
from pymongo import MongoClient
from itertools import combinations
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

client = MongoClient('...')
db = client['..']
ico=db['.']
colorlist=['#0072bd','#d95319','#edb120','#7e2f8e','#77ac30','#4dbeee','#a2142f']
colorlist=['#0072bd','#d95319','#edb120','#7e2f8e','lightblue','teal','lightgreen','pink','indigo','tomato','turquoise']
linecolors=['black','blue','red','yellow']

###ODDS###
def pandas_rsquare_for_every_attribute(G, collection, max, min,attr):
    attributes = [attr]
    i = 0
    for attribute in attributes:
        i += 1
        if attribute == 'degree':
            lista = cal.get_degree_frequency_list_for_bot_score(G, collection, max, min)
        elif attribute == 'hashtags' or attribute == 'urls' or attribute == 'mentionsAt':
            lista = cal.get_attribute_degree_frequency_list_for_bot_score(G, collection, attribute, max, min)
        elif attribute == 'node weights' or attribute == 'edge weights':
            lista = cal.get_weights_frequency_list(G, attribute)
        else:
            lista = cal.get_unique_attribute_degree_frequency_list(G,collection, attribute,max,min)
        df = pd.DataFrame(lista, columns=[attribute])
        total_users = df.shape[0]
        val = attribute
        stats_df = df \
            .groupby(val) \
            [val] \
            .agg('count') \
            .pipe(pd.DataFrame) \
            .rename(columns={val: 'frequency'})
        # PDF
        stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
        # CDF
        stats_df['cdf'] = stats_df['pdf'].cumsum()
        # CCDF
        stats_df['ccdf'] = 1 - stats_df['cdf']
        # ODDS
        stats_df = stats_df[:-1]
        stats_df['odds'] = stats_df['cdf'] / stats_df['ccdf']
        stats_df = stats_df.reset_index()
        # ODDS
        stats_df['odds'] += 1
        stats_df['copy'] = stats_df[val]
        stats_df[val] = pd.to_numeric(stats_df[val])
        stats_df[val] += 1
        stats_df = stats_df.dropna()
        stats_df['odds'] = stats_df['odds'].apply(np.log10)
        stats_df[val] = stats_df[val].apply(np.log10)
        stats_df = stats_df.loc[stats_df['odds'] >= 0]
        stats_df = stats_df.sort_values(by='odds', ascending=True)
        stats_df = stats_df[:-1]
        slope, intercept, r_value, p_value, std_err = stats.linregress(stats_df[val], stats_df['odds'])
    return (slope, r_value,intercept , total_users,stats_df,df)

def plot_odds_for_every_attribute(G,collection,max,min):
    attributes = ['urls', 'tweets', 'degree','retweets','hashtags','friends']
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    cols = 3
    rows = 2
    i = 0
    for attribute in tqdm(attributes):
        i += 1
        if attribute == 'degree':
            lista = cal.get_degree_frequency_list_for_bot_score(G, collection, max, min)
        elif attribute == 'hashtags' or attribute == 'urls':
            lista = cal.get_attribute_degree_frequency_list_for_bot_score(G, collection, attribute, max, min)
        elif attribute == 'node weights' or attribute == 'edge weights':
            lista = cal.get_weights_frequency_list(G, attribute)
        else:
            lista = cal.get_unique_attribute_degree_frequency_list(G, collection, attribute, max, min)
        df = pd.DataFrame(lista, columns=[attribute])
        # print (df)
        total_users = df.shape[0]
        val = attribute
        stats_df = df \
            .groupby(val) \
            [val] \
            .agg('count') \
            .pipe(pd.DataFrame) \
            .rename(columns={val: 'frequency'})
        # PDF
        stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
        # CDF
        stats_df['cdf'] = stats_df['pdf'].cumsum()
        # CCDF
        stats_df['ccdf'] = 1 - stats_df['cdf']
        # print (stats_df)
        # ODDS
        stats_df = stats_df[:-1]
        stats_df['odds'] = stats_df['cdf'] / stats_df['ccdf']
        stats_df = stats_df.reset_index()
        # ODDS
        stats_df['odds'] += 1
        stats_df[val] = pd.to_numeric(stats_df[val])
        stats_df[val] += 1
        stats_df = stats_df.dropna()
        stats_df['odds'] = stats_df['odds'].apply(np.log)
        stats_df[val] = stats_df[val].apply(np.log)
        stats_df = stats_df.loc[stats_df['odds'] >= 0]
        stats_df = stats_df.sort_values(by='odds', ascending=True)
        stats_df = stats_df[:-1]
        ax = plt.subplot(rows, cols, i)
        plt.ylabel('Odds Ratio')
        plt.xlabel('#'+attribute)
        slope, intercept, r_value, p_value, std_err = stats.linregress(stats_df[val], stats_df['odds'])
        line = slope * stats_df[val] + intercept
        plt.plot(stats_df[val], line,c='green', linewidth=10.0,alpha=0.25,
                 label='slope:' + str(round(slope, 2))+ '-R^2:' + str(round(r_value, 3)))
        plt.scatter(stats_df[val], stats_df['odds'], c=colorlist[i - 1], alpha=0.5, s=10, label=attribute)
        # plt.title(attribute + ' odds ratio - bot score in ['+str(min)+','+str(max)+']')
        plt.legend()
    # fig.canvas.set_window_title('Odds Ratio - total users:'+str(total_users))
    fig.tight_layout()
    plt.show()

def plot_odds_slope_for_every_attribute_over_time(Glist,collection,maxim,minim):
    attributes = ['hashtags', 'urls', 'tweets', 'degree']
    fontsize = 14
    fontsize_legend = 6
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    cols = 2
    rows = 2
    i = 0
    for attribute in tqdm(attributes):
        i += 1
        ax = plt.subplot(rows, cols, i)
        plt.title(attribute)
        valueList = []
        n=0
        for G in Glist:
            if attribute == 'degree':
                lista = cal.get_degree_frequency_list_for_bot_score(G, collection, maxim, minim)
            elif attribute == 'hashtags' or attribute == 'urls':
                lista = cal.get_attribute_degree_frequency_list_for_bot_score(G, collection, attribute, maxim, minim)
            elif attribute == 'node weights' or attribute == 'edge weights':
                lista = cal.get_weights_frequency_list(G, attribute)
            else:
                lista = cal.get_unique_attribute_degree_frequency_list(G, collection, attribute, maxim, minim)
            df = pd.DataFrame(lista, columns=[attribute])
            total_users = df.shape[0]
            val = attribute
            stats_df = df \
                .groupby(val) \
                [val] \
                .agg('count') \
                .pipe(pd.DataFrame) \
                .rename(columns={val: 'frequency'})
            # PDF
            stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
            # CDF
            stats_df['cdf'] = stats_df['pdf'].cumsum()
            # CCDF
            stats_df['ccdf'] = 1 - stats_df['cdf']
            # ODDS
            stats_df = stats_df[:-1]
            stats_df['odds'] = stats_df['cdf'] / stats_df['ccdf']
            stats_df = stats_df.reset_index()
            # ODDS
            stats_df['odds'] += 1
            stats_df[val] += 1
            stats_df = stats_df.dropna()
            stats_df['odds'] = stats_df['odds'].apply(np.log10)
            stats_df[val] = stats_df[val].apply(np.log10)
            stats_df = stats_df.loc[stats_df['odds'] >= 0]
            stats_df = stats_df.sort_values(by='odds', ascending=True)
            stats_df = stats_df[:-1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(stats_df[val], stats_df['odds'])
            valueList.append(slope)
            valueList.append(r_value)
            if n != len(Glist) - 1:
                plt.scatter(G.name[-10:],slope,c=colorlist[i-1],marker='o',s=100,label='_nolegend_')
                plt.scatter(G.name[-10:], r_value, c='black',marker='x',s=100,label='_nolegend_')
                plt.axhline(y=1, color='r', linestyle='-',label='_nolegend_')
            else:
                plt.scatter(G.name[-10:], slope, c=colorlist[i - 1], s=100,label='Slope for ' + attribute)
                plt.scatter(G.name[-10:], r_value, c='black',marker='x',s=100, label='R Square for ' + attribute)
                plt.axhline(y=1, color='r', linestyle='-', label='R square for perfect fit')
            # plt.ylim(min(valueList)-0.15,max(valueList)+0.25)
            plt.ylim(0.45, 1.8)
            plt.legend()
            plt.grid(color='grey', linestyle='-', linewidth=1)
            plt.xlabel('Date')
            plt.ylabel('Slope & R-Square')
            n+=1
    fig.canvas.set_window_title('Slope and R-Square over time')
    plt.grid(color='grey', linestyle='-', linewidth=1)
    plt.show()

def plot_rsquare_for_every_discrete_attribute(G,col, min,attr):
    df = pd.DataFrame(columns=['slope', 'r_value', 'total_users', 'limit'])
    slopeList = []
    r_values = []
    users = []
    limits = []
    for j in range(20, 530, 20):
        k = (j / 100)
        slope, r_value, total_users = pandas_rsquare_for_every_attribute(G, ekloges, k, min,attr)
        slopeList.append(slope)
        r_values.append(r_value)
        users.append(total_users)
        limits.append(k)
    df['slope'] = slopeList
    df['r_value'] = r_values
    df['total_users'] = users
    df['limit'] = limits
    print(df.head(1))
    pickle.dump(df, open('df_odds_hashtags_elections', 'wb'))
    df = pickle.load(open('df_odds_hashtags_elections', 'rb'))
    fig, ax = plt.subplots()
    sc = ax.scatter(df.limit, df.r_value, c=df.total_users,  cmap="Reds")
    plt.xlabel('max bot score')
    plt.ylabel('R square value')
    fig.colorbar(sc, ax=ax, label='number of users')
    plt.show()

def plot_rsquare_for_every_attribute_in_one_plot(G,col, min,attributes):
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    fig.canvas.set_window_title('Odds Ratio for all attributes')
    cols = 3
    rows = 2
    i = 0
    for a in tqdm(attributes):
        i += 1
        df = pd.DataFrame(columns=['slope', 'r_value', 'total_users', 'limit'])
        slopeList = []
        r_values = []
        users = []
        limits = []
        bx = plt.subplot(rows, cols, i)
        for j in range(20, 530, 20):
            k = (j / 100)
            slope, r_value,intercept , total_users,stats_df,df = pandas_rsquare_for_every_attribute(G, col, k, min,a)
            slopeList.append(slope)
            r_values.append(r_value)
            users.append(total_users)
            limits.append(k)
        df['slope'] = slopeList
        df['r_value'] = r_values
        df['total_users'] = users
        df['limit'] = limits
        print(df.head(1))
        # pickle.dump(df, open('df_odds_hashtags_elections', 'wb'))
        # df = pickle.load(open('df_odds_hashtags_elections', 'rb'))
        sc = plt.scatter(df.limit, df.r_value, c=df.total_users,  cmap="Reds")
        plt.xlabel('max bot score')
        plt.ylabel('R square value')
        plt.title(a)
        fig.colorbar(sc, label='number of users')
    plt.savefig('Faloutsos_Paper_Final_Images/ODDS_RSQUARE/'+str(col.name)+'_'+str(min)+'.jpeg',dpi=300)

def plot_rSQ_Slope_evolution(Glist,attributes):
    maxim=5.1
    minim=0
    fontsize = 14
    fontsize_legend = 8
    fig = plt.figure(figsize=(18, 9), facecolor='white')
    # fig.suptitle('Slope and R-square evolution over time for every attribute | botscore:['+str(minim)+','+str(maxim)+']', fontsize=16)
    cols = 4
    rows = 3
    i = 0
    for a in tqdm(attributes):
        i+=1
        slopeList=[]
        rlist=[]
        interList=[]
        dateList=np.array(range(1,6))
        for G in Glist:
            slope, r_value,intercept , total_users,stats_df,df = pandas_rsquare_for_every_attribute(G, ico, maxim, minim, a)
            slopeList.append(slope)
            rlist.append(r_value)
            interList.append(intercept)
        ax = plt.subplot(rows, cols, i)
        s, inter, r, p_, std_err = stats.linregress(dateList, slopeList)
        plt.scatter(dateList,slopeList,c=colorlist[attributes.index(a)],marker='o',s=100)
        line = s * dateList + inter
        plt.plot(dateList, line, c='blue', linewidth=2.0, linestyle='dashed',
                 label='slope:' + str(round(s, 2)) + ' - intercept:' + str(
                     round(inter, 2)) + ' - R_sq:' + str(round(r, 2)))
        plt.title(a)
        plt.xlabel('Dataset')
        plt.ylabel('slope value')
        plt.ylim(min(slopeList) - 0.4, max(slopeList) + 0.4)
        plt.legend()
        plt.grid()
        i+=1
        ax = plt.subplot(rows, cols, i)
        plt.scatter(dateList, rlist, c='red', marker='x', s=100)
        plt.axhline(y=1, color='purple', linestyle='-', label='R-square for perfect fit')
        plt.title(a)
        plt.xlabel('Dataset')
        plt.ylabel('R-Square value')
        plt.ylim(min(rlist) - 0.2, max(rlist) + 0.2)
        plt.legend()
        plt.grid()
    fig.tight_layout()
    plt.show()

def plot_rSQ_Slope_evolution_bots_nonbots(Glist,attributes):
    maxim=2.9
    minim=0
    fontsize = 14
    fontsize_legend = 8
    fig = plt.figure(figsize=(18, 9), facecolor='white')
    # fig.suptitle('Slope and R-square evolution over time for every attribute | botscore:['+str(minim)+','+str(maxim)+']', fontsize=16)
    cols = 4
    rows = 3
    i = 0
    for a in tqdm(attributes):
        i+=1
        slopeList=[]
        rlist=[]
        interList=[]
        dateList=np.array(range(1,6))
        slopeList2 = []
        rlist2 = []
        interList2 = []
        for G in Glist:
            slope, r_value,intercept , total_users,stats_df,df = pandas_rsquare_for_every_attribute(G, ico, maxim, minim, a)
            slopeList.append(slope)
            rlist.append(r_value)
            interList.append(intercept)

            slope2, r_value2, intercept2, total_users2, stats_df2, df2 = pandas_rsquare_for_every_attribute(G, ico, 5.1,4.9, a)
            slopeList2.append(slope2)
            rlist2.append(r_value2)
            interList2.append(intercept2)

        ax = plt.subplot(rows, cols, i)
        s, inter, r, p_, std_err = stats.linregress(dateList, slopeList)
        plt.scatter(dateList,slopeList,c='green',marker='o',s=100,label='nonBots:'+str(total_users))
        line = s * dateList + inter
        plt.scatter(dateList, slopeList2, c='red', marker='o', s=100,label='Bots:'+str(total_users2))
        plt.plot(dateList, line, c='green', linewidth=2.0, linestyle='dashed',
                 label='Non Bots:['+str(minim)+','+str(maxim)+']')
        s, inter, r, p_, std_err = stats.linregress(dateList, slopeList2)
        line = s * dateList + inter
        plt.plot(dateList, line, c='red', linewidth=2.0, linestyle='dashed',
                 label='Bots:[4.9,5.1]')
        plt.title(a)
        plt.xlabel('Dataset')
        plt.ylabel('slope value')
        plt.ylim(min(slopeList) - 0.2, max(slopeList2) + 0.2)
        plt.legend()
        plt.grid()
        i+=1
        ax = plt.subplot(rows, cols, i)
        plt.scatter(dateList, rlist, c='green', marker='x', s=100,label='Non Bots')
        plt.scatter(dateList, rlist2, c='red', marker='x', s=100,label='Bots')
        plt.axhline(y=1, color='purple', linestyle='-', label='R-square for perfect fit')
        plt.title(a)
        plt.xlabel('Dataset')
        plt.ylabel('R-Square value')
        plt.ylim(min(rlist) - 0.2, max(rlist) + 0.2)
        plt.legend()
        plt.grid()
    fig.tight_layout()
    fontsize_legend = 8
    plt.show()


###AIS###
def plot_ais_fit_for_every_attribute(G,attributes,botmin,botmax,evolving=False):
    # df = toolset.load_ais_data_in_pandas(attribute,'final_ico',G,botmin,botmax)
    # df=df[['volume','label']]
    # df = df.sort_values(by='volume',ascending=False)
    # print (df.head(100))
    attributePairs=[('volume','mass'),('mass','triangles'),('volume','triangles'),('gcc','volume'),('mass','gcc'),('triangles','smax')]
    # attributePairs = [('volume', 'mass')]
    fontsize = 14
    fontsize_legend = 8
    fig = plt.figure(figsize=(18, 9), facecolor='white')
    fig.tight_layout()
    # fig.suptitle('Patterns of AIS graph features',fontsize=16)
    cols = 3
    rows = 2
    i=0
    for attribute in attributes:
        for pair in attributePairs:
            print (pair)
            df1 = toolset.load_ais_data_in_pandas(attribute,'final_ico',G,0,5.1)
            # df1 = toolset.load_ais_data_in_pandas(attribute,'final_ico',G,botmin,botmax)
            if evolving==True:
                if attribute=='urls':
                    df1=df1[df1['label'].isin(pickle.load(open('urlsEvolved','rb')))]
                else:
                    df1 = df1[df1['label'].isin(pickle.load(open('tagsEvolved', 'rb')))]
            # if pair==('triangles','smax'):
            df1=df1[df1['triangles']>0]
            x=pair[0]
            y=pair[1]
            df1[x]+=1
            # if x!='density':
            df1[y]+=1
            df1[x] = df1[x].apply(np.log10)
            df1[y] = df1[y].apply(np.log10)
            ax = plt.subplot(rows, cols, i+1)
            plt.scatter(df1[x], df1[y], c='silver', alpha=0.75, label='correlation:'+str(round(stats.pearsonr(df1[x], df1[y])[0],2)))
            plt.title(str(pair) + ' : ' + attribute)
            try:
                line = toolset.log_fit_line(df1[x], df1[y])
                plt.scatter(line['x'], line['bin_avg'], color='k', marker='d', s=100)
                # plt.plot(line['x'], line['y'], color='k', linestyle='--', linewidth=4)
                slope, intercept, r_value, p_value, std_err = stats.linregress(line['x'], line['bin_avg'])
                lineb = slope * line['x'] + intercept
                plt.plot(line['x'], lineb, c='k',
                         label=' fit slope:' + str(round(slope, 2)) + '-R square:' + str(round(r_value, 2)))
            except:
                # continue
                slope, intercept, r_value, p_value, std_err = stats.linregress(df1[x], df1[y])
                line = slope * df1[x] + intercept
                plt.plot(df1[x], line, c='k',marker='d',markevery=200,markersize=10,
                         label=' fit slope:' + str(round(slope, 2)) + '-R square:' + str(round(r_value, 2)))
                # plt.plot(df1[x], line, c='orange',
                #          label=' fit slope,intercept: ' + str(round(slope, 3)) + ' ' + str(round(intercept, 2)))
            # plt.plot(line['x'], line['y'], color='green', linestyle='--', linewidth=4,label=line['m'])
            plt.legend(fontsize='x-small')
            plt.xlabel(x)
            plt.ylabel(y)
            i+=1
    fig.tight_layout()
    plt.show()

def plot_ais_fit_for_every_attribute_bots(G,attributes,botmin1,botmin2,botmax1,botmax2,evolving=False):
    # df = toolset.load_ais_data_in_pandas(attribute,'final_ico',G,botmin,botmax)
    # df=df[['volume','label']]
    # df = df.sort_values(by='volume',ascending=False)
    # print (df.head(100))
    attributePairs=[('volume','mass'),('mass','triangles'),('volume','triangles'),('gcc','volume'),('mass','gcc'),('triangles','smax')]
    # attributePairs = [('mass','triangles')]
    fontsize = 14
    fontsize_legend = 8
    fig = plt.figure(figsize=(18, 9), facecolor='white')
    # fig = plt.figure()
    fig.tight_layout()
    # fig.suptitle('Patterns of AIS graph features',fontsize=16)
    cols = 3
    rows = 4
    i=0
    for attribute in attributes:
        for pair in attributePairs:
            print (pair)
            df1 = toolset.load_ais_data_in_pandas(attribute,'final_ico',G,botmin1,botmin2)
            df2 = toolset.load_ais_data_in_pandas(attribute,'final_ico',G,botmax1,botmax2)
            if evolving==True:
                if attribute=='urls':
                    df1=df1[df1['label'].isin(pickle.load(open('urlsEvolved','rb')))]
                    df2 = df2[df2['label'].isin(pickle.load(open('urlsEvolved', 'rb')))]
                else:
                    df1 = df1[df1['label'].isin(pickle.load(open('tagsEvolved', 'rb')))]
                    df2 = df2[df2['label'].isin(pickle.load(open('tagsEvolved', 'rb')))]
            # if pair==('triangles','smax'):
            df1=df1[df1['triangles']>0]
            df2 = df2[df2['triangles'] > 0]
            x=pair[0]
            y=pair[1]
            df1[x]+=1
            # if x!='density':
            df1[y]+=1
            df1[x] = df1[x].apply(np.log10)
            df1[y] = df1[y].apply(np.log10)

            df2[x] += 1
            df2[y] += 1
            df2[x] = df2[x].apply(np.log10)
            df2[y] = df2[y].apply(np.log10)
            ax = plt.subplot(rows, cols, i+1)
            plt.scatter(df1[x], df1[y], c='green', alpha=0.15, label='NonBots:correlation:'+str(round(stats.pearsonr(df1[x], df1[y])[0],2)))
            plt.scatter(df2[x], df2[y], c='red', alpha=0.15,
                        label='Bots:correlation:' + str(round(stats.pearsonr(df2[x], df2[y])[0], 2)))
            plt.title(str(pair) + ' : ' + attribute)
            try:
                line = toolset.log_fit_line(df1[x], df1[y])
                plt.scatter(line['x'], line['bin_avg'], color='orange', marker='d', s=10)
                # plt.plot(line['x'], line['y'], color='k', linestyle='--', linewidth=4)
                slope, intercept, r_value, p_value, std_err = stats.linregress(line['x'], line['bin_avg'])
                lineb = slope * line['x'] + intercept
                plt.plot(line['x'], lineb, c='orange',linestyle='dashdot',
                         label='Non-bots fit slope:' + str(round(slope, 2)) + '-R square:' + str(round(r_value, 2)))
            except:
                # continue
                slope, intercept, r_value, p_value, std_err = stats.linregress(df1[x], df1[y])
                line = slope * df1[x] + intercept
                plt.plot(df1[x], line, linestyle='dashdot',c='k',marker='d',markevery=200,markersize=10,
                         label=' fit slope:' + str(round(slope, 2)) + '-R square:' + str(round(r_value, 2)))
            try:
                line = toolset.log_fit_line(df2[x], df2[y])
                plt.scatter(line['x'], line['bin_avg'], color='k', marker='d', s=10)
                # plt.plot(line['x'], line['y'], color='k', linestyle='--', linewidth=4)
                slope, intercept, r_value, p_value, std_err = stats.linregress(line['x'], line['bin_avg'])
                lineb = slope * line['x'] + intercept
                plt.plot(line['x'], lineb, c='k',linestyle='dashdot',
                         label='Bots fit slope:' + str(round(slope, 2)) + '-R square:' + str(round(r_value, 2)))
            except:
                # continue
                slope, intercept, r_value, p_value, std_err = stats.linregress(df2[x], df2[y])
                line = slope * df1[x] + intercept
                plt.plot(df2[x], line, c='k', marker='d', markevery=200, markersize=10,linestyle='dashdot',
                         label=' fit slope:' + str(round(slope, 2)) + '-R square:' + str(round(r_value, 2)))
                # plt.plot(df1[x], line, c='orange',
                #          label=' fit slope,intercept: ' + str(round(slope, 3)) + ' ' + str(round(intercept, 2)))
            # plt.plot(line['x'], line['y'], color='green', linestyle='--', linewidth=4,label=line['m'])
            plt.legend(fontsize='x-small')
            plt.xlabel(x)
            plt.ylabel(y)
            i+=1
    fig.tight_layout()
    plt.show()

def slope_rsquare_for_every_attribute_AIS(G, ico, botmin1,botmin2,botmax1, botmax2, attribute, pair,evolving=False):
    df1 = toolset.load_ais_data_in_pandas(attribute, 'final_ico', G, botmin1, botmin2)
    df1 = df1[df1['triangles'] > 0]
    if evolving == True:
        if attribute == 'urls':
            df1 = df1[df1['label'].isin(pickle.load(open('urlsEvolved', 'rb')))]
        else:
            df1 = df1[df1['label'].isin(pickle.load(open('tagsEvolved', 'rb')))]
    x = pair[0]
    y = pair[1]
    df1[x] += 1
    df1[y] += 1
    df1[x] = df1[x].apply(np.log10)
    df1[y] = df1[y].apply(np.log10)
    try:
        line = toolset.log_fit_line(df1[x], df1[y])
        corr = round(stats.pearsonr(df1[x], df1[y])[0],2)
        users = df1.shape[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(line['x'], line['bin_avg'])
    except:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df1[x], df1[y])
        users = df1.shape[0]
        corr = round(stats.pearsonr(df1[x], df1[y])[0], 2)
    df1 = toolset.load_ais_data_in_pandas(attribute, 'final_ico', G, botmax1, botmax2)
    df1 = df1[df1['triangles'] > 0]
    x = pair[0]
    y = pair[1]
    df1[x] += 1
    df1[y] += 1
    df1[x] = df1[x].apply(np.log10)
    df1[y] = df1[y].apply(np.log10)
    try:
        line = toolset.log_fit_line(df1[x], df1[y])
        corr2 = round(stats.pearsonr(df1[x], df1[y])[0],2)
        users2 = df1.shape[0]
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(line['x'], line['bin_avg'])
    except:
        slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(df1[x], df1[y])
        users2 = df1.shape[0]
        corr2 = round(stats.pearsonr(df1[x], df1[y])[0], 2)
    return slope, r_value , corr , slope2 , r_value2 , corr2, users , users2

def slope_rsquare_for_every_attribute_AIS_all(G, ico, botmin,botmax, attribute, pair,evolving=False):
    df1 = toolset.load_ais_data_in_pandas(attribute, 'final_ico', G, botmin,botmax)
    if evolving == True:
        if attribute == 'urls':
            df1 = df1[df1['label'].isin(pickle.load(open('urlsEvolved', 'rb')))]
        else:
            df1 = df1[df1['label'].isin(pickle.load(open('tagsEvolved', 'rb')))]
    x = pair[0]
    y = pair[1]
    df1[x] += 1
    df1[y] += 1
    df1[x] = df1[x].apply(np.log10)
    df1[y] = df1[y].apply(np.log10)
    try:
        line = toolset.log_fit_line(df1[x], df1[y])
        corr = round(stats.pearsonr(df1[x], df1[y])[0],2)
        users = df1.shape[0]
        slope, intercept, r_value, p_value, std_err = stats.linregress(line['x'], line['bin_avg'])
    except:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df1[x], df1[y])
        users = df1.shape[0]
        corr = round(stats.pearsonr(df1[x], df1[y])[0], 2)
    return slope, r_value , corr , users , intercept , df1[x]

def plot_rSQ_Slope_evolution_for_AIS(Glist,attribute,botmin1,botmin2,botmax1,botmax2,features,evolving=False):
    fontsize = 14
    fontsize_legend = 8
    fig = plt.figure(figsize=(18, 9), facecolor='white')
    cols = 4
    rows = 3
    i = 0
    for f in tqdm(features):
        i+=1
        slopeList=[]
        slopeList2 = []
        corrlist=[]
        corrlist2=[]
        userlist=[]
        userlist2=[]
        rlist=[]
        rlist2=[]
        dateList=np.array(range(1,5))
        for G in Glist:
            slope, r_value , corr , slope2 , r_value2 , corr2,users,users2 = slope_rsquare_for_every_attribute_AIS(G, ico, botmin1,botmin2, botmax1,botmax2, attribute,f,evolving)
            slopeList.append(slope)
            userlist.append(users)
            userlist2.append(users2)
            slopeList2.append(slope2)
            rlist.append(r_value)
            rlist2.append(r_value2)
            corrlist.append(corr)
            corrlist2.append(corr2)
        ax = plt.subplot(rows, cols, i)
        s, inter, r, p_, std_err = stats.linregress(dateList, slopeList)
        plt.scatter(dateList,slopeList,c='green',marker='o',s=100)
        line = s * dateList + inter
        plt.plot(dateList, line, c='green', linewidth=2.0, linestyle='dashed',
                 label='nonBots')
        for j, txt in enumerate(userlist):
            plt.annotate(txt, (dateList[j], slopeList[j]))
        s, inter, r, p_, std_err = stats.linregress(dateList, slopeList2)
        plt.scatter(dateList, slopeList2, c='red', marker='o', s=100)
        for j, txt in enumerate(userlist2):
            plt.annotate(txt, (dateList[j], slopeList2[j]))
        line = s * dateList + inter
        plt.plot(dateList, line, c='red', linewidth=2.0, linestyle='dashed',
                 label='bots')
        plt.title('slope evolution for '+str(f))
        plt.xlabel('Dataset')
        plt.ylabel('slope value')
        plt.ylim(min(slopeList2) - 0.2, max(slopeList) + 0.2)
        plt.legend(fontsize='x-small')
        plt.grid()
        i+=1
        ax = plt.subplot(rows, cols, i)
        plt.scatter(dateList, corrlist, c='green', marker='x', s=100,label='NonBots Corr')
        # plt.scatter(dateList, rlist, c='green', marker='^', s=100, label='NonBots R^2')
        plt.axhline(y=1, color='purple', linestyle='-')
        plt.scatter(dateList, corrlist2, c='red', marker='x', s=100,label='Bots Corr')
        # plt.scatter(dateList, rlist2, c='red', marker='^', s=100, label='Bots R^2')
        plt.title('Correlation/R-square evolution for '+str(f))
        plt.xlabel('Dataset')
        plt.ylabel('Correlation/R-square value')
        plt.ylim(min(corrlist2) - 0.2, max(corrlist) + 0.2)
        plt.legend(fontsize='x-small')
        plt.grid()
    fig.tight_layout()
    plt.show()

def plot_rSQ_Slope_evolution_for_AIS_all_scores(Glist,attribute,botmin,botmax,features,evolving=False):
    fontsize = 14
    fontsize_legend = 8
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    # fig.suptitle('Slope and R-square evolution over time for every AIS graph feature | botscore:['+str(botmin)+','+str(botmax)+']', fontsize=16)
    cols = 4
    rows = 3
    i = 0
    for f in tqdm(features):
        i+=1
        slopeList=[]
        corrlist=[]
        userlist=[]
        rlist=[]
        dateList=np.array(range(1,5))
        for G in Glist:
            slope, r_value , corr , users , intercept , dff = slope_rsquare_for_every_attribute_AIS_all(G, ico, botmin,botmax, attribute,f,evolving)
            slopeList.append(slope)
            userlist.append(users)
            corrlist.append(corr)
            rlist.append(r_value)
        ax = plt.subplot(rows, cols, i)
        s, inter, r, p_, std_err = stats.linregress(dateList, slopeList)
        plt.scatter(dateList,slopeList,c=colorlist[features.index(f)],marker='o',s=100)
        line = s * dateList + inter
        plt.plot(dateList, line, c=colorlist[features.index(f)], linewidth=2.0, linestyle='dashed',
                 label='all')
        # for j, txt in enumerate(userlist):
        #     plt.annotate(txt, (dateList[j], slopeList[j]))
        plt.title('slope evolution for '+str(f))
        plt.xlabel('Dataset')
        plt.ylabel('slope value')
        plt.ylim(min(slopeList) - 0.2, max(slopeList) + 0.2)
        plt.legend()
        plt.grid()
        i+=1
        ax = plt.subplot(rows, cols, i)
        plt.scatter(dateList, corrlist, c=colorlist[features.index(f)], marker='x', s=100,label='correlation')
        plt.scatter(dateList, rlist, c='black',
                    marker='^', s=100, label='R-square')
        plt.axhline(y=1, color='purple', linestyle='-',label='perfect fit')
        plt.title('Correlation evolution for '+str(f))
        plt.xlabel('Dataset')
        plt.ylabel('Correlation value')
        plt.ylim(min(corrlist) - 0.2, max(corrlist) + 0.2)
        plt.legend(fontsize='x-small')
        plt.grid()
    fig.tight_layout()
    plt.show()

def pdf_cdf_ccdf_offs_AIS(pairs,distro,attribute,val):
    for p in pairs:
        df = toolset.load_ais_data_in_pandas(attribute,'final_ico',G,p[0],p[1])
        shape = df.shape[0]
        print (df.columns)
        stats_df = df \
            .groupby(val) \
            [val] \
            .agg('count') \
            .pipe(pd.DataFrame) \
            .rename(columns={val: 'frequency'})

        # PDF
        stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
        print(stats_df.head(10))
        # CDF
        stats_df['cdf'] = stats_df['pdf'].cumsum()
        print(stats_df.head(10))
        # CCDF
        stats_df['ccdf'] = 1 - stats_df['cdf']
        print(stats_df.head(10))
        # ODDS
        # stats_df = stats_df[:-1]
        stats_df['odds'] = stats_df['cdf'] / stats_df['ccdf']
        print(stats_df.head(10))
        stats_df = stats_df.reset_index()
        # print(stats_df.head(10))
        # ODDS
        plt.plot(stats_df[val],stats_df[distro],label=(p,shape))
        plt.legend()
        plt.xscale('log')
        plt.xlabel(val)
        plt.ylabel(distro)
        # plt.yscale('log')
    plt.show()





###COMMUNITY?###
def community_plots():
    comm = pickle.load(open('louvain_community_bipartite_hashtags_cumulative_11_12_2018','rb'))
    df = toolset.load_node_data_in_pandas('final_ico',G)
    df['community'] = df['label'].map(comm)
    df['bot_norm'] = (df['botscore']-df['botscore'].min())/(df['botscore'].max()-df['botscore'].min())
    dff = df.groupby(["community"]).size().reset_index(name="Time")
    dff = dff.sort_values(by='Time',ascending=False)
    print (dff.head(10))
    top25 = dff['community'].tolist()[:25]
    slopelist=[]
    botlist=[]
    rlist=[]
    userlist=[]
    comlist=[]
    for c in top25:
        x='tweets'
        y='hashtags'
        # print (df.columns)
        df1 = df[df['community']==c]
        # plt.scatter(df1[x],df1[y])
        slope, intercept, r_value, p_value, std_err = stats.linregress(df1[x], df1[y])
        if r_value>0.90:
            comlist.append(c)
            userlist.append(df1.shape[0])
            rlist.append(r_value)
            botlist.append(sum(df1['bot_norm'].tolist())/len(df1['bot_norm'].tolist()))
    plt.scatter(comlist,rlist)
    for j, txt in enumerate(botlist):
        plt.annotate(txt, (comlist[j], rlist[j]))
    plt.show()
    for c in comlist:
        df1=df[df['community']==c]
        df1[x]+=1
        df1[y]+=1
        df1[x] = df1[x].apply(np.log10)
        df1[y] = df1[y].apply(np.log10)
        plt.scatter(df1[x],df1[y],label=c)
        slope, intercept, r_value, p_value, std_err = stats.linregress(df1[x], df1[y])
        line = slope * df1[x] + intercept
        plt.plot(df1[x],line,c='black')
    plt.show()

###EVOLVING GRAPHS###
def get_evolving_df(Glist):
    df1 = toolset.load_ais_data_in_pandas('urls','final_ico',Glist[1],0,5.1).set_index('label')
    print (df1.index.is_unique)
    df2 = toolset.load_ais_data_in_pandas('urls','final_ico',Glist[2],0,5.1).set_index('label')
    df2 = df2.loc[~df2.index.duplicated(keep='first')]
    df3 = toolset.load_ais_data_in_pandas('urls','final_ico',Glist[3],0,5.1).set_index('label')
    df3 = df3.loc[~df3.index.duplicated(keep='first')]
    df4 = toolset.load_ais_data_in_pandas('urls','final_ico',Glist[4],0,5.1).set_index('label')
    df4 = df4.loc[~df4.index.duplicated(keep='first')]
    df = pd.concat([df1, df2,df3,df4],axis='columns',keys=['1','2','3','4'])
    cols = ['volume','mass','triangles','gcc','smax']
    for c in cols:
        df[('5',c)]=df[('2',c)]-df[('1',c)]
        df[('6',c)]=df[('3',c)]-df[('2',c)]
        df[('7',c)]=df[('4', c)]-df[('3', c)]
    pickle.dump(df,open('df_evolution','wb'))
    cols = ['volume','mass','triangles','gcc','smax']
    df = pickle.load(open('df_evolution','rb'))
    # df=df[(df[('5','volume')]>0) and (df[('6','volume')]>0) and (df[('7','volume')]>0)]
    df = df.loc[(df[('1','volume')]>0) & (df[('5','volume')]>0) & (df[('6','volume')]>0) & (df[('7','volume')]>0)]
    df=df[[('5','volume'),('6','volume'),('7','volume')]]
    print (df.shape)
    urlsEvolved = df.index.values.tolist()
    print (urlsEvolved)
    pickle.dump(urlsEvolved,open('urlsEvolved','wb'))

def plot_evolution_of_diameter(Glist):
    evolving = pickle.load(open('tagsEvolved', 'rb'))
    diameterChange = {}
    botdiameterChange = {}
    datasets=[0,5,10,15]
    for e in tqdm(evolving):
        notBotdiameterList = []
        notBotsingletonList = []
        for G in Glist:
            df = toolset.load_ais_data_with_diameter('hashtags','final_ico',G,0,2.1)
            df=df[df['label']==e]
            if df.shape[0]>0:
                xval = df['diameter'].values[0]
                yval = df['singletons'].values[0]
                notBotdiameterList.append(xval)
                notBotsingletonList.append(yval)
            else:
                continue
        if len(notBotdiameterList)>1:
            diameterChange[e]=notBotdiameterList[-1]-notBotdiameterList[0]
        else:
            diameterChange[e]=0
        print (e,diameterChange[e])
    pickle.dump(diameterChange,open('notbotdiameterchange','wb'))
    for e in tqdm(evolving):
        botdiameterList = []
        botsingletonList = []
        for G in Glist:
            df = toolset.load_ais_data_with_diameter('hashtags','final_ico',G,4,5.2)
            df=df[df['label']==e]
            if df.shape[0]>0:
                xval = df['diameter'].values[0]
                yval = df['singletons'].values[0]
                botdiameterList.append(xval)
                botsingletonList.append(yval)
            else:
                continue
        if len(botdiameterList)>1:
            botdiameterChange[e]=botdiameterList[-1]-botdiameterList[0]
        else:
            botdiameterChange[e]=0
        print (e,botdiameterChange[e])
    pickle.dump(botdiameterChange,open('botdiameterchange','wb'))
        # plt.plot(datasets,diameterList)
        # plt.show()
    for G in Glist:
        df = toolset.load_ais_data_with_diameter('hashtags', 'final_ico', G, 0,2.1)
        df=df[df['label'].isin(evolving)]
        df['diameterchange'] = df['label'].map(diameterChange)
        # df['diameter']+=1
        # df['botscore']+=1
        plt.scatter(df['botscore'],df['diameterchange'],c='green')
        df = toolset.load_ais_data_with_diameter('hashtags', 'final_ico', G,4,5.1)
        df = df[df['label'].isin(evolving)]
        df['diameterchange'] = df['label'].map(botdiameterChange)
        plt.scatter(df['botscore'], df['diameterchange'],c='red')
        # plt.xscale('log')
        # plt.yscale('log')
        plt.show()

def plot_evolution_of_evolving(g):
    x = 'mass'
    y = 'triangles'
    df = toolset.load_ais_data_in_pandas('hashtags', 'final_ico', g, 0, 5.1)
    df1=df[df['botscore']>4.0]
    bot = df1['label'].tolist()
    df2 = df[df['botscore'] <= 2.1]
    nbot = df2['label'].tolist()
    # bot = ['losangeles','hollywood','bmw','cryptology','follow4follow','coinanalystico','service','cryptonityex','manager','reality']
    # nbot = ['dogecoinmoon','cryptoart','tokes','moonsoon','dashconference','ignis','netcoins','dashentrepreneur','telfam','freespeech']
    slope, r_value, corr, users, intercept,dfx = slope_rsquare_for_every_attribute_AIS_all(g, ico, 0,5.1, 'hashtags', (x,y),evolving=True)
    line = slope * dfx + intercept
    plt.plot(dfx, line, c='orange', linewidth=2.0, linestyle='dashed',
             label=g.name)
    for b in bot:
        xlist=[]
        ylist=[]
        for G in Glist[1:]:
            try:
                df = toolset.load_ais_data_in_pandas('hashtags', 'final_ico', G, 0, 5.1)
                df = df[df['label']==b]
                df[x] += 1
                df[y] += 1
                df[x] = df[x].apply(np.log10)
                df[y] = df[y].apply(np.log10)
                xval = df[x].values[0]
                yval = df[y].values[0]
                xlist.append(xval)
                ylist.append(yval)
            except:
                pass
        plt.plot(xlist,ylist,c='red')
    for b in nbot:
        xlist=[]
        ylist=[]
        for G in Glist[1:]:
            df = toolset.load_ais_data_in_pandas('hashtags', 'final_ico', G, 0, 5.1)
            df = df[df['label']==b]
            df[x] += 1
            df[y] += 1
            df[x] = df[x].apply(np.log10)
            df[y] = df[y].apply(np.log10)
            xval = df[x].values[0]
            yval = df[y].values[0]
            xlist.append(xval)
            ylist.append(yval)
        plt.plot(xlist,ylist,c='green')
    plt.show()

def plot_degree_distros_of_subgraphs():
    telcoin = nx.read_edgelist("cryptonity.csv")
    doge = nx.read_edgelist("altcoin.csv")
    doged = dict(sorted(dict(nx.degree(doge)).items()))
    telcod = dict(sorted(dict(nx.degree(telcoin)).items()))
    # print(telcod)
    ax = plt.subplot(1, 2, 1)
    print ((list(doged.values())))
    from collections import Counter
    print (Counter((list(doged.values()))))
    data = np.log10((list(doged.values())))
    print (Counter(data))
    # print (data)
    # print (len(data))
    # print (list)
    from scipy.stats import norm
    mean, std = norm.fit(data)
    plt.hist(data, bins=100, color=colorlist[0], density=True, label='altcoin')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    # print (y)
    plt.xlabel('Degree - Log10 Scale')
    plt.ylabel('Number of nodes - frequency')
    plt.plot(x, y)
    plt.legend()
    plt.yticks([])
    # plt.show()
    #
    ax = plt.subplot(1, 2, 2)
    data = np.log10(list(telcod.values()))
    # print (len(data))
    from scipy.stats import norm
    mean, std = norm.fit(data)
    plt.hist(data, bins=100, color=colorlist[1], density=True, label='cryptonity')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    y = norm.pdf(x, mean, std)
    plt.xlabel('Degree - Log10 Scale')
    plt.ylabel('Number of nodes - frequency')
    plt.plot(x, y)
    plt.legend()
    plt.yticks([])
    plt.show()

#CLASSIFIERS#
def prepare_data(attribute,featurelist):
    G = Glist[-1]
    df_not_bot = toolset.load_ais_data_in_pandas(attribute, 'final_ico', G, 0, 2.5)
    df_bot = toolset.load_ais_data_in_pandas(attribute, 'final_ico', G, 2.5, 5.1)
    notbots = [0 for i in df_not_bot['botscore']]
    df_not_bot['botlabel']=notbots
    bots = [1 for i in df_bot['botscore']]
    df_bot['botlabel']=bots
    df = pd.concat([df_bot,df_not_bot])
    df=df[df['triangles']>0]
    # print (df.head(3000))
    X = df[featurelist]  # Features
    y = df.botlabel
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)
    return X_train, X_test, y_train, y_test,X,y

def classify(X_train, X_test, y_train, y_test,X,y):
    seed = 42
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('MLP',MLPClassifier()))
    models.append(('RF',RandomForestClassifier()))
    models.append(('ADA',AdaBoostClassifier()))
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(model, X, y, cv=kfold,
                                                     scoring=scoring)

        filename = name+'.sav'
        pickle.dump(model,open('models/'+filename,'wb'))
        # import time
        # print (cv_results)
        # time.sleep(10)

        results.append(cv_results.mean())
        names.append(name)
        # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        # print(msg)
    # boxplot algorithm comparison
    return results,names

def visualize_classifiers():
    fig = plt.figure()
    fig.suptitle('Algorithm comparison')
    features = ['volume', 'mass', 'triangles', 'gcc', 'smax','fraudarNorm']
    # features = ['volume', 'mass',  'fraudarNorm']
    featureCombos = list(combinations(features,6))
    print (featureCombos)
    print (len(featureCombos))
    rows = 1
    columns = 1
    i=0
    for combo in featureCombos:
        # print (list(f))
        a,b,c,d,e,f = prepare_data('hashtags',[combo[0],combo[1],combo[2],combo[3] , combo[4],combo[5] ])
        results,names = classify(a,b,c,d,e,f)
        i+=1
        ax = plt.subplot(rows, columns, i)
        ax.set_xticklabels(names)
        plt.ylabel('Accuracy')
        plt.title(str(combo[0])+'-'+str(combo[1])+'-'+str(combo[2])+'-'+str(combo[3])+'-'+str(combo[4]+'-'+str(combo[5])))
        maximum = max(results)
        print (colorlist[i-1])
        plt.bar(names,results,color=colorlist[i-1],label = str(round(maximum,2))+':'+ str(names[results.index(maximum)]))
        plt.legend()
        plt.yticks(np.arange(0,maximum,step=0.25))
    fig.tight_layout()
    plt.show()

def multiplyList(myList):
    # Multiply elements one by one
    result = 1
    for x in myList:
        result = result*(1-x)
    return result

def sumList(myList):
    result = 0
    for x in myList:
        if x==1:
            result=result+np.log10(0.01)
        else:
            result = result + np.log10((1 - x))
    return result

def toss_coin(n,p):
    n=1
    a = np.random.binomial(n,p)
    return a











