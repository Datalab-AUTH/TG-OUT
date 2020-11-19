from collections import Counter
import matplotlib.pyplot as plt
import toolset
import math
import pandas as pd
import numpy as np
import scipy
from scipy import stats
import calculations as cal
from scipy.stats import scoreatpercentile
from sklearn import linear_model
import pickle

colorlist=['#0072bd','#d95319','#edb120','#7e2f8e','#77ac30','#4dbeee','a2142f']

'''TAKES AS INPUT LIST OF GRAPHS, ATTRIBUTE (SOCIAL,HASHTAGS,URLS) AND TYPO OF DISTRIBUTION (PDF,CDF,CCDF,ODDS
RETUNS AN IMAGE OF EACH DISTRIBUTION RESPECTFULLY'''
def temporal_multiplot(glist, attribute, typo, collection):
    colorlist=['#0072bd','#d95319','#edb120','#7e2f8e','#77ac30','4dbeee','a2142f']
    j=0
    for G in glist:
        if attribute == 'social':
            lista = cal.get_degree_frequency_list(G)
        elif attribute == 'hashtags' or attribute == 'urls':
            lista = cal.get_attribute_degree_frequency_list(G, attribute)
        else:
            lista = cal.get_unique_attribute_degree_frequency_list(G, attribute)
        listar = np.asarray(lista)
        data = np.bincount(listar)
        s = float(data.sum())
        cdf = data.cumsum(0) / s

        if typo == 'pdf':
            plt.plot(range(len(data)), data, alpha=1, label=G.name[-10:],c=colorlist[j])
            plt.yscale('log')
            plt.xscale('log')
            plt.ylabel('Freq')
            plt.xlabel('Degree')
            plt.title(attribute + ' PDF')
            plt.legend()
            plt.savefig(collection.name+'/'+typo+'/' + attribute + '_temporal_pdf_plot.png', dpi=(300))
            # plt.show()

        if typo == 'cdf':
            # plt.clf()
            s = float(data.sum())
            cdf = data.cumsum(0) / s
            plt.plot(range(len(cdf)), cdf, alpha=1, label=G.name[-10:],c=colorlist[j])
            plt.xscale('log')
            plt.ylim([0, 1])
            plt.ylabel('CDF')
            plt.xlabel('Degree')
            plt.legend()
            plt.title(attribute + ' CDF')
            plt.savefig(collection.name+'/'+typo+'/' + attribute + '_temporal_cdf_plot.png', dpi=(300))
            # plt.show()

        if typo == 'ccdf':
            # plt.clf()
            ccdf = 1 - cdf
            plt.plot(range(len(ccdf)), ccdf, alpha=1, label=G.name[-10:],c=colorlist[j])
            plt.xscale('log')
            plt.yscale('log')
            plt.ylim([0, 1])
            plt.ylabel('CCDF')
            plt.title(attribute + ' CCDF')
            plt.xlabel('Degree')
            plt.legend()
            plt.savefig(collection.name+'/' +typo+'/'+ attribute + '_temporal_ccdf_plot.png', dpi=(300))
            # plt.show()

        if typo == 'odds':
            ccdf = 1 - cdf
            odds = cdf / ccdf
            nprange = np.arange(len(odds))
            logA = np.log10(nprange + 1)
            logB = np.log10(odds + 1)
            print(odds.shape, nprange.shape)
            plt.plot(logA, logB, label=G.name[-10:],c=colorlist[j])
            plt.ylabel('Odds Ratio')
            plt.title(attribute + ' odds ratio')
            plt.legend()
            plt.savefig(collection.name+'/' +typo+'/'+ attribute + '_temporal_odds_ratio_plot.png', dpi=(300))
            # plt.show()
        j += 1
    plt.clf()

'''TAKES AS INPUT A SPECIFIC GRAPH AND RETURNS ALL THE DEGREE DISTRIBUTIONS SEPARATELY FOR EACH GRAPH - ATTRIBUTE'''
def multiplot(G,attribute,collection):
    if attribute=='social':
        lista = cal.get_degree_frequency_list(G)
    elif attribute=='hashtags' or attribute=='urls' or attribute=='mentionsAt':
        lista = cal.get_attribute_degree_frequency_list(G, attribute)
    elif attribute=='node weights' or attribute=='edge weights':
        lista=cal.get_weights_frequency_list(G,attribute)
    else:
        lista = cal.get_unique_attribute_degree_frequency_list(G,attribute)
    listar = np.asarray(lista)
    print (listar)
    data = np.bincount(listar)
    """ Plot Distribution """
    plt.plot(range(len(data)),data,c='orange',alpha=1,label='pdf')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Freq')
    plt.xlabel('Degree')
    plt.title(attribute+' PDF')
    plt.savefig(collection.name+'_'+G.name+'_'+attribute+'_pdf_plot.png', dpi=(300))
    plt.clf()

    """ Plot CDF """
    s = float(data.sum())
    cdf = data.cumsum(0)/s
    plt.plot(range(len(cdf)),cdf,c='orange',alpha=1,label='cdf')
    plt.xscale('log')
    plt.ylim([0,1])
    plt.ylabel('CDF')
    plt.xlabel('Degree')
    plt.title(attribute+' CDF')
    plt.savefig(collection.name+'_'+G.name+'_'+attribute+'_cdf_plot.png', dpi=(300))
    plt.clf()

    """ Plot CCDF """
    ccdf = 1-cdf
    plt.plot(range(len(ccdf)),ccdf,c='orange',alpha=1,label='ccdf')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([0,1])
    plt.ylabel('CCDF')
    plt.title(attribute+' CCDF')
    plt.xlabel('Degree')
    plt.savefig(collection.name+'_'+G.name+'_'+attribute+'_ccdf_plot.png', dpi=(300))
    plt.clf()

    """Plot ODDS Ratio"""
    ccdf = 1 - cdf
    odds = cdf / ccdf
    nprange = np.arange(len(odds))
    logA = np.log10(nprange + 1)
    logB = np.log10(odds + 1)
    plt.plot(logA, logB, c='red',alpha=1,label='odds ratio')
    plt.ylabel('Odds Ratio')
    plt.title(attribute + ' odds ratio')
    plt.legend()
    plt.savefig(collection.name+'_'+G.name+'_'+attribute + '_odds_ratio_plot.png', dpi=(300))
    plt.clf()

'''TAKES AS INPUT A SPECIFIC GRAPH AND RETURNS ALL THE DEG DISTROS USING PANDAS!'''
def pandas_multiplot(G,collection,max,min):
    attributes = ['hashtags', 'urls', 'tweets', 'social','friends','followers']
    # fontsize = 14
    # fontsize_legend = 11
    # fig = plt.figure(figsize=(12, 9), facecolor='white')
    # fig.canvas.set_window_title('Odds Ratio for all attributes')
    # cols = 3
    # rows = 2
    i = 0
    for attribute in attributes:
        i+=1
        if attribute=='social':
            lista = cal.get_degree_frequency_list_for_bot_score(G,collection,max,min)
        elif attribute=='hashtags' or attribute=='urls' or attribute=='mentionsAt':
            lista = cal.get_attribute_degree_frequency_list_for_bot_score(G,collection,attribute,max,min)
        elif attribute=='node weights' or attribute=='edge weights':
            lista=cal.get_weights_frequency_list(G,attribute)
        else:
            lista = cal.get_unique_attribute_degree_frequency_list(G,attribute)
        df = pd.DataFrame(lista, columns =[attribute])
        total_users = df.shape[0]
        val=attribute
        stats_df = df \
        .groupby(val) \
        [val] \
        .agg('count') \
        .pipe(pd.DataFrame) \
        .rename(columns = {val: 'frequency'})
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
        ax = plt.subplot(rows, cols, i)

        # CDF
        plt.plot(stats_df[val], stats_df['cdf'], label='cdf-bot score:'+str(min)+'-'+str(max))
        plt.xscale('log')
        plt.ylim([0, 1])
        plt.ylabel('CDF')
        plt.xlabel('Degree')
        plt.title(attribute + ' CDF')
        plt.legend()
        plt.show()
        # # PDF
        plt.plot(stats_df[val],stats_df['pdf'],label='pdf-bot score:'+str(min)+'-'+str(max))
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel('Freq')
        plt.xlabel('Degree')
        plt.title(attribute + ' PDF')
        plt.legend()
        plt.show()

        # ODDS
        stats_df['odds']+=1
        stats_df[val]+=1
        stats_df=stats_df.dropna()
        stats_df['odds']=stats_df['odds'].apply(np.log10)
        stats_df[val]=stats_df[val].apply(np.log10)
        stats_df=stats_df.loc[stats_df['odds']>=0]
        stats_df = stats_df.sort_values(by='odds',ascending=True)
        stats_df = stats_df[:-1]
        plt.scatter(stats_df[val], stats_df['odds'],c=colorlist[i-1],alpha=0.5, label='bot_score: '+str(min)+'-'+str(max))
        plt.scatter(stats_df[val], stats_df['odds'],c=colorlist[i-1],alpha=0.5, label=attribute)
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel('Odds Ratio')
        plt.xlabel('#'+attribute)
        slope, intercept, r_value, p_value, std_err = stats.linregress(stats_df[val],stats_df['odds'])
        line = slope * stats_df[val] + intercept
        plt.plot(stats_df[val], line,c='black', linewidth=2.0, linestyle='dashed',
                 label='slope:' + str(round(slope, 2)) + '-intercept:' + str(
                     round(intercept, 2)) + '-R^2:' + str(round(r_value, 3)))
        plt.title(attribute + ' odds ratio')
        plt.legend()
    # plt.show()
    return (slope,r_value,total_users)

'''TAKES AS INPUT SPECIFIC GRAPH AND SPECIFIC BOT SCORE, RETURNS ALL THE ANNOTATED (BOT SCORE) DEGREE DISTROS'''
def multiplot_BOT(G,attribute,collection_string,col,min_bot_score,max_bot_score):
    if attribute=='social':
        lista = cal.get_degree_frequency_list_for_bot_score(G,col,max_bot_score,min_bot_score)
    elif attribute=='hashtags' or attribute=='urls' or attribute=='mentionsAt':
        lista = cal.get_attribute_degree_frequency_list_for_bot_score(G,col,attribute,max_bot_score,min_bot_score)
    elif attribute=='node weights' or attribute=='edge weights':
        lista=cal.get_weights_frequency_list(G,attribute)
    else:
        lista = cal.get_unique_attribute_degree_frequency_list(G,col,attribute,max_bot_score,min_bot_score)
    listar = np.asarray(lista)
    print (listar)
    data = np.bincount(listar)
    """ Plot Distribution """
    plt.plot(range(len(data)),data,c='orange',alpha=1,label='pdf')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Freq')
    plt.xlabel('Degree')
    plt.title(attribute+' PDF')
    plt.savefig(collection_string+'_'+G.name+'_'+attribute+'_pdf_plot_for_bot_score_between_'+
                str(min_bot_score)+'_'+str(max_bot_score)+'.png', dpi=(300))
    plt.clf()

    """ Plot CDF """
    s = float(data.sum())
    cdf = data.cumsum(0)/s
    plt.plot(range(len(cdf)),cdf,c='orange',alpha=1,label='cdf')
    plt.xscale('log')
    plt.ylim([0,1])
    plt.ylabel('CDF')
    plt.xlabel('Degree')
    plt.title(attribute+' CDF')
    plt.savefig(collection_string+'_'+G.name+'_'+attribute+'_cdf_plot_for_bot_score_between_'+
                str(min_bot_score)+'_'+str(max_bot_score)+'.png', dpi=(300))
    plt.clf()

    """ Plot CCDF """
    ccdf = 1-cdf
    plt.plot(range(len(ccdf)),ccdf,c='orange',alpha=1,label='ccdf')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([0,1])
    plt.ylabel('CCDF')
    plt.title(attribute+' CCDF')
    plt.xlabel('Degree')
    plt.savefig(collection_string+'_'+G.name+'_'+attribute+'_ccdf_plot_for_bot_score_between_'+
                str(min_bot_score)+'_'+str(max_bot_score)+'.png', dpi=(300))
    plt.clf()

    """Plot ODDS Ratio"""
    ccdf = 1 - cdf
    odds = cdf / ccdf
    nprange = np.arange(len(odds))
    logA = np.log10(nprange + 1)
    logB = np.log10(odds + 1)
    plt.plot(logA, logB, c='red',alpha=1,label='odds ratio')
    plt.ylabel('Odds Ratio')
    plt.title(attribute + ' odds ratio')
    plt.legend()
    plt.savefig(collection_string+'_'+G.name+'_'+attribute + '_odds_ratio_plot_for_bot_score_between_'+
                str(min_bot_score)+'_'+str(max_bot_score)+'.png', dpi=(300))
    plt.clf()

'''TAKES AS INPUT SPECIFIC GRAPH AND SPECIFIC BOT SCORE, RETURNS ALL THE ANNOTATED (BOT SCORE) DEGREE DISTROS AND COMPARES BETWEEN THEM'''
def multiplot_BOT_comparison(G,attribute,collection_string,col,min_bot_score1,min_bot_score_2,max_bot_score1,max_bot_score_2):
    if attribute=='social':
        lista = cal.get_degree_frequency_list_for_bot_score(G,col,max_bot_score1,max_bot_score_2)
        lista2 = cal.get_degree_frequency_list_for_bot_score(G,col,min_bot_score1,min_bot_score_2)
    elif attribute=='hashtags' or attribute=='urls' or attribute=='mentionsAt':
        lista = cal.get_attribute_degree_frequency_list_for_bot_score(G,col,attribute,max_bot_score_2,max_bot_score1)
        lista2 = cal.get_attribute_degree_frequency_list_for_bot_score(G, col,attribute,min_bot_score_2,min_bot_score1)
    elif attribute=='node weights' or attribute=='edge weights':
        lista=cal.get_weights_frequency_list(G,attribute)
    else:
        lista = cal.get_unique_attribute_degree_frequency_list(G,col,attribute,max_bot_score_2,max_bot_score1)
        lista2 = cal.get_unique_attribute_degree_frequency_list(G, col,attribute,min_bot_score_2,min_bot_score1)
    listar = np.asarray(lista)
    print (listar)
    listar2 = np.asarray(lista2)
    data = np.bincount(listar)
    data2 = np.bincount(listar2)
    """ Plot Distribution """
    plt.plot(range(len(data)),data,c='red',alpha=1,label='pdf bots')
    plt.plot(range(len(data2)), data2, c='green', alpha=1, label='pdf not bots')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Freq')
    plt.xlabel('Degree')
    plt.title(attribute+' PDF')
    plt.legend()
    plt.savefig(collection_string+'_'+G.name+'_'+attribute+'_pdf_plot_for_bot_score_between_'+
                str(min_bot_score1)+'_'+str(max_bot_score_2)+'.png', dpi=(300))
    plt.clf()

    """ Plot CDF """
    s = float(data.sum())
    cdf = data.cumsum(0)/s
    plt.plot(range(len(cdf)),cdf,c='red',alpha=1,label='cdf bots')
    s = float(data2.sum())
    cdf = data2.cumsum(0) / s
    plt.plot(range(len(cdf)), cdf, c='green', alpha=1, label='cdf not bots')
    plt.xscale('log')
    plt.ylim([0,1])
    plt.ylabel('CDF')
    plt.xlabel('Degree')
    plt.title(attribute+' CDF')
    plt.legend()
    plt.savefig(collection_string+'_'+G.name+'_'+attribute+'_cdf_plot_for_bot_score_between_'+
                str(min_bot_score1)+'_'+str(max_bot_score_2)+'.png', dpi=(300))
    plt.clf()

    """ Plot CCDF """
    s = float(data.sum())
    cdf = data.cumsum(0) / s
    ccdf = 1-cdf
    plt.plot(range(len(ccdf)),ccdf,c='red',alpha=1,label='ccdf bots')
    s = float(data2.sum())
    cdf = data2.cumsum(0) / s
    ccdf = 1 - cdf
    plt.plot(range(len(ccdf)), ccdf, c='green', alpha=1, label='ccdf not bots')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([0,1])
    plt.ylabel('CCDF')
    plt.title(attribute+' CCDF')
    plt.xlabel('Degree')
    plt.savefig(collection_string+'_'+G.name+'_'+attribute+'_ccdf_plot_for_bot_score_between_'+
                str(min_bot_score1)+'_'+str(max_bot_score_2)+'.png', dpi=(300))
    plt.clf()

    """Plot ODDS Ratio"""
    s = float(data.sum())
    cdf = data.cumsum(0) / s
    ccdf = 1 - cdf
    odds = cdf / ccdf
    nprange = np.arange(len(odds))
    logA = np.log10(nprange + 1)
    logB = np.log10(odds + 1)
    plt.plot(logA, logB, c='red',alpha=1,label='odds ratio bots')
    s = float(data2.sum())
    cdf = data2.cumsum(0) / s
    ccdf = 1 - cdf
    odds = cdf / ccdf
    nprange = np.arange(len(odds))
    logA = np.log10(nprange + 1)
    logB = np.log10(odds + 1)
    plt.plot(logA, logB, c='green', alpha=1, label='odds ratio not bots')
    plt.ylabel('Odds Ratio')
    plt.title(attribute + ' odds ratio')
    plt.legend()
    plt.savefig(collection_string+'_'+G.name+'_'+attribute + '_odds_ratio_plot_for_bot_score_between_'+
                str(min_bot_score1)+'_'+str(max_bot_score_2)+'.png', dpi=(300))
    plt.clf()

def get_total_distribution(G,attribute,collection_string,col,min_bot_score1,min_bot_score_2,max_bot_score1,max_bot_score_2):
    if attribute == 'social':
        lista = cal.get_degree_frequency_list_for_bot_score(G, col,
                                                            max_bot_score1,
                                                            max_bot_score_2)
        lista2 = cal.get_degree_frequency_list_for_bot_score(G, col,
                                                             min_bot_score1,
                                                             min_bot_score_2)
    elif attribute == 'hashtags' or attribute == 'urls' or attribute == 'mentionsAt':
        lista = cal.get_attribute_degree_frequency_list_for_bot_score(G, col,
                                                                      attribute,
                                                                      max_bot_score_2,
                                                                      max_bot_score1)
        lista2 = cal.get_attribute_degree_frequency_list_for_bot_score(G, col,
                                                                       attribute,
                                                                       min_bot_score_2,
                                                                       min_bot_score1)
    elif attribute == 'node weights' or attribute == 'edge weights':
        lista = cal.get_weights_frequency_list(G, attribute)
    else:
        lista = cal.get_unique_attribute_degree_frequency_list(G, col,
                                                               attribute,
                                                               max_bot_score_2,
                                                               max_bot_score1)
        lista2 = cal.get_unique_attribute_degree_frequency_list(G, col,
                                                                attribute,
                                                                min_bot_score_2,
                                                                min_bot_score1)
    dfa = pd.DataFrame(lista,columns=['obj'])
    val = 'obj'
    stats_df = dfa \
        .groupby(val) \
        [val] \
        .agg('count') \
        .pipe(pd.DataFrame) \
        .rename(columns={val: 'frequency'})
    stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
    # CDF
    stats_df['cdf'] = stats_df['pdf'].cumsum()
    # CCDF
    stats_df['ccdf'] = 1 - stats_df['cdf']
    # ODDS
    stats_df['odds'] = stats_df['cdf'] / stats_df['ccdf']
    stats_df = stats_df.reset_index()
    print (stats_df.head(10))

    dfb = pd.DataFrame(lista2, columns=['obj'])
    val = 'obj'
    stats_df_b = dfb \
        .groupby(val) \
        [val] \
        .agg('count') \
        .pipe(pd.DataFrame) \
        .rename(columns={val: 'frequency'})
    stats_df_b['pdf'] = stats_df_b['frequency'] / sum(stats_df_b['frequency'])
    # CDF
    stats_df_b['cdf'] = stats_df_b['pdf'].cumsum()
    # CCDF
    stats_df_b['ccdf'] = 1 - stats_df_b['cdf']
    # ODDS
    stats_df_b['odds'] = stats_df_b['cdf'] / stats_df_b['ccdf']
    stats_df_b = stats_df_b.reset_index()
    print(stats_df_b.head(10))
    return stats_df,stats_df_b


def oddsAllAttributes(G):
    attributes = ['hashtags','urls','tweets','social']
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    fig.canvas.set_window_title('Odds Ratio for all attributes')
    cols = 2
    rows = 2
    i=0
    for a in attributes:
        i+=1
        if a=='social':
            lista = cal.get_degree_frequency_list(G)
        elif a=='hashtags' or a=='urls' or a=='mentionsAt':
            lista = cal.get_attribute_degree_frequency_list(G, a)
        elif a=='node weights' or a=='edge weights':
            lista=cal.get_weights_frequency_list(G,a)
        else:
            lista = cal.get_unique_attribute_degree_frequency_list(G,a)
        listar = np.asarray(lista)
        print (listar)
        data = np.bincount(listar)
        s = float(data.sum())
        cdf = data.cumsum(0) / s
        cdf = cdf[:-1].copy()
        ccdf = 1 - cdf
        odds = cdf / ccdf
        nprange = np.arange(len(odds))
        print (nprange)
        xAxis = np.log10(nprange+1)
        yAxis = np.log10(odds+1)
        ax = plt.subplot(rows, cols, i)
        plt.scatter(xAxis, yAxis, c=colorlist[i-1], marker='+', label=a)
        slope, intercept, r_value, p_value, std_err = stats.linregress(xAxis, yAxis)
        line = slope * xAxis + intercept
        plt.plot(xAxis, line, c='black', linewidth=2.0, linestyle='dashed',
                 label='slope:' + str(round(slope, 3)) + ' - intercept:' + str(
                     round(intercept, 3)) + ' - Rsquare:' + str(round(r_value, 3)))
        ax.set_ylim((-0.25, 5))
        plt.legend()
    plt.show()

'''FUNCTION THAT PLOTS ODDS RATIO LINE FITTING USING RANSAC FOR ATTRIBUTE TO ATTRIBUTE'''
def line_fit_odds_ratio(glist,attribute):
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    fig.canvas.set_window_title('Odds Ratio for: '+attribute)
    cols = 3
    rows = 2
    print (rows)
    # i = 320
    # j = 0
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'lime', 'cyan', 'yellow', 'aqua', 'magenta',
              'fuchsia']
    slopeList = []
    interList = []
    i=0
    for G in glist:
        i+=1
        if attribute=='social':
            lista = cal.get_degree_frequency_list(G)
        elif attribute=='hashtags' or attribute=='urls':
            lista = cal.get_attribute_degree_frequency_list(G, attribute)
        else:
            lista = cal.get_unique_attribute_degree_frequency_list(G,attribute)
        listar = np.asarray(lista)
        data = np.bincount(listar)
        s = float(data.sum())
        cdf = data.cumsum(0) / s
        cdf=cdf[:-1].copy()
        ccdf = 1 - cdf
        odds = cdf / ccdf
        nprange = np.arange(len(odds))
        logA = np.log10(nprange + 1)
        logB = np.log10(odds + 1)

        ax = plt.subplot(rows, cols, i)
        plt.scatter(logA, logB,c='orange', marker='+', label='odds_ratio')
        xAxis = np.reshape(logA, (-1, 1))
        yAxis = np.reshape(logB, (-1, 1))
        # print (type(xAxis))
        # print (xAxis.shape[0])
        ransac = linear_model.RANSACRegressor(min_samples=xAxis.shape[0],max_trials=1000)
        ransac.fit(xAxis, yAxis)
        line_y_ransac = ransac.predict(xAxis)
        slope = (line_y_ransac[1] - line_y_ransac[0]) / (xAxis[1] - xAxis[0])
        intercept = line_y_ransac[1] - slope * xAxis[1]
        r_value, p_value = scipy.stats.pearsonr(xAxis, line_y_ransac)
        if type(slope) == np.ndarray:
            slope = slope[0]
        if type(intercept) == np.ndarray:
            intercept = intercept[0]
        if type(r_value) == np.ndarray:
            r_value = r_value[0]

        slopeList.append((G.name, slope))
        interList.append((G.name, intercept))
        plt.plot(xAxis, line_y_ransac, c='blue',linestyle='dashed',linewidth=2.0, label='slope:' + str(round(slope, 3)) + ' - intercept:' + str(
            round(intercept, 3)) + ' - Rsquare:' + str(round(r_value, 3)))
        plt.xlabel(attribute)
        plt.ylabel('odds ratio')
        plt.title('Odds Ratio for: '+attribute+' of '+G.name)
        plt.legend()

    ymin=min(n[1] for n in interList)-1
    ymax=max(n[1] for n in interList)+1
    xmin=0
    xmax=(max(n[1] for n in slopeList)+1)
    print (ymin,ymax,xmin,xmax)
    i+=1
    ax = plt.subplot(rows, cols, i)
    plt.scatter(*zip(*slopeList),marker='x',s=100, c='black')
    ax.grid()
    plt.xticks(rotation=45)
    plt.xlabel('Day')
    plt.ylabel('Slope')
    ax.set_ylim((xmin, xmax))
    i += 1
    ax = plt.subplot(rows, cols, i)
    plt.scatter(*zip(*interList),marker ='x',s=100, c='purple')
    ax.grid()
    ax.set_ylim((ymin, ymax))
    plt.xlabel('Day')
    plt.ylabel('intercept')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(attribute + '_ransac_fitting.png')
    # plt.show()
    plt.clf()

'''FUNCTION THAT PLOTS ODDS RATIO LINE FITTING USING SIMPLE LIN REGRESS FOR ATTRIBUTE TO ATTRIBUTE'''
def line_fit_odds_ratio_withour_ransac(glist,attribute):
    fontsize = 14
    fontsize_legend = 11
    fig = plt.figure(figsize=(18, 10), facecolor='white')
    fig.canvas.set_window_title('Odds Ratio for: '+attribute)
    cols = 3
    rows = 2
    print(rows)
    colors = ['black', 'blue', 'green', 'orange', 'purple', 'lime', 'cyan', 'yellow', 'aqua', 'magenta',
              'fuchsia']
    slopeList = []
    interList = []
    i=0
    dataset=''
    if len(glist)==4:
        dataset = 'elections'
    else:
        dataset = 'ico'
        glist= glist[1:]
    fig.suptitle(dataset+'-'+attribute)
    for G in glist:
        i+=1
        if attribute == 'social':
            lista = cal.get_degree_frequency_list(G)
        elif attribute == 'hashtags' or attribute == 'urls':
            lista = cal.get_attribute_degree_frequency_list(G, attribute)
        else:
            lista = cal.get_unique_attribute_degree_frequency_list(G, attribute)
        listar = np.asarray(lista)
        data = np.bincount(listar)
        s = float(data.sum())
        cdf = data.cumsum(0) / s
        cdf = cdf[:-1].copy()
        ccdf = 1 - cdf
        odds = cdf / ccdf
        nprange = np.arange(len(odds))
        xAxis = np.log10(nprange + 1)
        yAxis = np.log10(odds + 1)
        pickle.dump(xAxis,open('xaxis','wb'))
        pickle.dump(yAxis,open('yxaxis','wb'))


        ax = plt.subplot(rows, cols, i)
        plt.scatter(xAxis, yAxis,c='orange', marker='+', label='odds_ratio')

        #IF I WANT TO BREAK IT INTO A CERTAIN POINT AND DRAW TWO FITTING LINES
        length = (250)
        xAxisx = xAxis[length:]
        # print (xAxisx)
        yAxisx = yAxis[length:]
        xAxisy = xAxis[0:length]
        # print (xAxisy)
        yAxisy = yAxis[0:length]
        slope, intercept, r_value, p_value, std_err = stats.linregress(xAxisx, yAxisx)
        slopeList.append((G.name, slope))
        interList.append((G.name, intercept))
        line = slope * xAxisx + intercept
        slopeList.append((G.name, slope))
        interList.append((G.name, intercept))
        plt.plot(xAxisx, line, c='blue', linewidth=2.0, linestyle='dashed',
                 label='slope:' + str(round(slope, 3)) + ' - intercept:' + str(
                     round(intercept, 3)) + ' - Rsquare:' + str(round(r_value, 3)))
        slope, intercept, r_value, p_value, std_err = stats.linregress(xAxisy, yAxisy)
        line = slope * xAxisy + intercept
        plt.plot(xAxisy, line, c='cyan', linewidth=2.0, linestyle='dashed',
                 label='slope:' + str(round(slope, 3)) + ' - intercept:' + str(
                     round(intercept, 3)) + ' - Rsquare:' + str(round(r_value, 3)))
        plt.legend()

        #IF I WANT ONE FITTING LINE
        # slope, intercept, r_value, p_value, std_err = stats.linregress(xAxis, yAxis)
        # slopeList.append((G.name[-10:], slope))
        # interList.append((G.name[-10:], intercept))
        # line = slope * xAxis + intercept
        # plt.plot(xAxis, line, c='blue', linewidth=2.0, linestyle='dashed',
        #                   label='slope:' + str(round(slope, 3)) + ' - intercept:' + str(
        #                       round(intercept, 3)) + ' - Rsquare:' + str(round(r_value, 3)))
        # plt.xlabel(attribute)
        # plt.ylabel('odds ratio')
        # plt.title(G.name[-10:])
        # plt.legend()
        # plt.savefig('odds_ratio/'+G.name+'_odds_ratio_'+attribute+'.png')
        # plt.xscale('log')
        # plt.show()
    ymin=min(n[1] for n in interList)-1
    ymax=max(n[1] for n in interList)+1
    xmin=0
    xmax=(max(n[1] for n in slopeList)+1)
    print (ymin,ymax,xmin,xmax)
    i+=1
    ax = plt.subplot(rows, cols, i)
    plt.scatter(*zip(*slopeList),marker='x',s=100, c='black')
    ax.grid()
    plt.xlabel('Day')
    plt.ylabel('Slope')
    plt.xticks(rotation=30)
    ax.set_ylim((-5, 5))
    i+=1
    ax = plt.subplot(rows, cols, i)
    plt.scatter(*zip(*interList),marker ='x',s=100, c='purple')
    ax.grid()
    plt.xlabel('Day')
    plt.ylabel('intercept')
    plt.xticks(rotation=30)
    ax.set_ylim((-5, 5))
    # plt.tight_layout()
    plt.savefig(dataset+'_odds_ratio_threshold_' + attribute + '_fitting.png')
    # plt.show()
    plt.clf()

'''FUNCTION THAT PLOTS ATTRIBUTE TO ATTRIBUTE FILES FOR SPECIFIC GRAPH'''
def draw_attribute_to_attribute_for_specific_graph(x,y,type,Gname):
    df = toolset.load_node_data_in_pandas(type,Gname)
    df[x]+=1
    df[y]+=1
    plt.scatter(x=df[x], y=df[y], alpha=0.3, s=50, label=Gname)
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(fontsize = 'x-small')
    plt.title(x + ' - ' + y + ' Distribution for ' + Gname)
    # plt.savefig('plots/' + x + ' - ' + y + ' Distribution for ' + G.name +'.png', dpi=(300))
    plt.show()
    plt.clf()

'''FUNCTION THAT PLOTS ATTRIBUTE TO ATTRIBUTE FOR SPECIFIV GRAPH USING MAPPING DICTS'''
def draw_attribute_to_attribute_with_mapping(x,y,type,mapping,G):
    df= toolset.load_node_data_in_pandas(type,G.name)
    df[x]+=1
    df[y]+=1
    colors = ['red', 'blue', 'green', 'cyan', 'orange', 'black', 'brown', 'lime', 'pink', 'purple']
    df['mapping'] = df['label'].map(mapping)
    unknown = df[df['mapping']==0]
    plt.scatter(unknown[x], unknown[y], c=colors[5], alpha=0.3, s=50, label='bot score = 0')
    plt.show()
    for i in range(0,5):
        mask=df[(df['mapping']>i) & (df['mapping']<=i+1)]
        plt.scatter(mask[x],mask[y],c=colors[i],alpha=0.3, s=50, label='bot score >'+str(i)+' <'+str(i+1))
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(fontsize='x-small')
        plt.show()

'''FUNCTION THAT PLOTS ATTRIBUTE TO ATTRIBUTE FILES FOR ALL GRAPHS'''
def draw_attribute_to_attribute_for_all_graphs(Glist,x,y,type):
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    cols=2
    rows = 2
    colors = [ 'blue', 'green', 'orange', 'black', 'purple', 'brown', 'lime', 'pink', 'red']
    slopeList=[]
    i=0
    if len(Glist)>4:
        Glist=Glist[1:]
        dataset = 'ico'
    else:
        dataset = 'elections'
    fig.suptitle(x + '-' + y + ' distribution for '+dataset+' dataset' )
    for G in Glist:
        i+=1
        df = toolset.load_node_ico_data_in_pandas(type, G.name)
        df[x] += 1
        df[y] += 1
        ax = plt.subplot(rows, cols, i)
        plt.scatter(x=df[x], y=df[y], alpha=0.3, s=50,c=colors[Glist.index(G)], label=G.name)
        # slope, intercept, r_value, p_value, std_err = stats.linregress(df[x], df[y])
        # slopeList.append((G.name,slope))
        # line = slope * df[x] + intercept
        # plt.scatter(df[x], line, c='red',
        #         label=' fit slope,intercept: ' + str(round(slope, 3)) + ' ' + str(round(intercept, 2)))
        plt.legend(fontsize='x-small')
        plt.grid(color='grey', linestyle='-', linewidth=0.2)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.xscale('log')
        plt.yscale('log')
        # plt.title( x + ' - ' + y + '_Distribution')
        plt.legend( fontsize = 'x-small')
    # ax = plt.subplot(rows, cols,len(Glist) + 1)
    # ax.scatter(*zip(*slopeList), marker='x', s=100, c='black')
    # ax.grid()
    # plt.xticks(rotation=45)
    # plt.xlabel('Day')
    # plt.ylabel('Slope')
    # plt.show()
    plt.savefig(dataset + ' ' + x + ' - ' + y +'_'+ type+'_Distribution.png', dpi=(300))

'''FUNCTION THAT PLOTS AIS ATTRIBUTES FILES FOR SPECIFIC GRAPH'''
def draw_ais_attribute_for_specific_graph(x,y,attribute,G):
    df = toolset.load_ais_data_in_pandas(attribute, G.name)
    df[x] += 1
    if y != 'density':
        df[y] += 1
    colors = ['red', 'blue', 'green', 'cyan', 'orange', 'black', 'brown', 'lime', 'pink', 'purple','red', 'blue', 'green', 'cyan', 'orange', 'black', 'brown', 'lime', 'pink', 'purple','magenta','red']
    allattributes = pickle.load(open('allattributes', 'rb'))
    atdict = {}
    for a in allattributes:
        atdict[str(allattributes.index(a))] = len(a)
    df['length']=df['label'].map(atdict)
    print (df.head(5))
    length = list(set(df['length'].tolist()))
    print (length)
    cols = 4
    rows = math.ceil(len(length[:10]) / cols)
    print (rows,cols)
    # import time
    # time.sleep(5)
    df[x] = df[x].apply(np.log10)
    df[y] = df[y].apply(np.log10)
    # x = np.asarray(df[x])
    # y = np.asarray(df[y])
    # logA = np.log10(x)
    # logB = np.log10(y)
    slopeList=[]
    for l in length[:10]:
        mask = df[df['length']==l]
        print ('index',length.index(l)+1)
        ax = plt.subplot(rows, cols, length.index(l)+1)
        ax.scatter(mask[x],mask[y],c=colors[length.index(l)],label=str(l)+' common')
        slope, intercept, r_value, p_value, std_err = stats.linregress(mask[x], mask[y])
        slopeList.append(slope)
        line = slope * mask[x] + intercept
        ax.plot(mask[x], line, c='red',
                 label=' fit slope,intercept: ' + str(round(slope, 3)) + ' ' + str(round(intercept, 2)))
        plt.legend(fontsize='x-small')
    ax=plt.subplot(rows, cols, len(length[:10])+1)
    ax.scatter(length[:10],slopeList,label = 'slope evolvement')
        # print (mask.head(5))
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.xlabel(x)
    plt.ylabel(y)
    # plt.xscale('log')
    # if y == 'density':
    #     plt.yscale('linear')
    # else:
    #     plt.yscale('log')
    plt.legend(fontsize='x-small')
    # plt.title(x + ' - ' + y + ' Distribution for ' + G.name)
    # plt.savefig('plots/AIS/' + x + ' - ' + y + ' Distribution for ' + G.name + '.png', dpi=(300))
    # plt.clf()
    plt.show()

'''FUNCTION THAT PLOTS AIS ATTRIBUTE FILES FOR ALL GRAPHS - OVERLAPPING'''
def draw_ais_attribute_for_all_graphs(x,y,Glist,attribute,dataset):
    for G in Glist:
        df = toolset.load_ais_data_in_pandas(attribute,dataset,G)
        print (df.columns)
        df[x] += 1
        df[y] += 1
        if y!='density':
            df[y] += 1
            plt.xscale('linear')
            plt.yscale('linear')
        fig, ax = plt.subplots()
        sc = ax.scatter(df[x], df[y], c=df['botscore'], cmap="Reds")
        fig.colorbar(sc, ax=ax, label='bot score')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
        plt.scatter(x=df[x], y=df[y],c=colorlist[0], alpha=0.3, s=50, label=G.name[-10:])
        plt.grid(color='grey', linestyle='-', linewidth=0.2)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(x + ' - ' + y + '_Distribution for '+attribute)
        plt.legend(fontsize='x-small')
        plt.show()
    # plt.savefig('plots/AIS/all_together/' + x + ' - ' + y + '_' + attribute + '_Distribution.png', dpi=(300))

