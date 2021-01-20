import networkx as nx
from pymongo import MongoClient
import pymongo
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta,datetime
import pickle
from tqdm import tqdm
from itertools import combinations
from random import randint

client = MongoClient('...')
db = client['..']
col=db['.']

#GLOBAL VARIABLE THAT DEFINES THE DELTA
PER_DAYS=5

'''GETS THE DATE OF THE OLDEST & NEWEST DOCUMENT - USED TO START CREATING THE TEMPORAL GRAPHS
RETURNS datetime.date objects'''
def get_date_of_oldest_newest_document(col):
    iterator = col.find().sort('created_at',1)
    oldest_doc_date = (iterator[0]['created_at'])
    iterator = col.find().sort('created_at', -1)
    newest_doc_sate = (iterator[0]['created_at'])
    return (oldest_doc_date,newest_doc_sate)

# def get_temporal_weighted_graphs(col):
#     G=nx.Graph()
#     start,end = get_date_of_oldest_newest_document(col)
#     # print (start,end)
#     until = start + timedelta(days=PER_DAYS)
#     until = datetime.combine(until.date(), datetime.min.time())
#     while until.date() <= end.date():
#         users=set()
#         notInDb=set()
#         it = col.find({"created_at": {"$lte": until}})
#         for k in it:
#             userid = k['user']['id_str']
#             users.add(userid)
#         print('users in our DB until: '+str(until.strftime('%m_%d_%Y'))+' are: '+ str(len(users)))
#         G.add_nodes_from(list(users))
#         iterator = col.find({"created_at": {"$lte": until,"$gte":start}})
#         for i in iterator:
#             user = i['user']['id_str']
#             entities = i['entities']
#             if 'user_mentions' in entities:
#                 for m in entities['user_mentions']:
#                     if G.has_edge(user, m['id_str']):
#                         G[user][m['id_str']]['weight'] += 1.0
#                     else:
#                         G.add_edge(user, m['id_str'], weight=1.0)
#                         if m['id_str'] not in users:
#                             notInDb.add(m['id_str'])
#         G.remove_edges_from(G.selfloop_edges())
#         pickle.dump(G, open('graphs/weighted_' + str(until.strftime('%m_%d_%Y')), 'wb'))
#         pickle.dump(users, open('users/allUsers_until_' + str(until.strftime('%m_%d_%Y')), 'wb'))
#         until = until + timedelta(days=PER_DAYS)
#         print('users in our DB:', len(users))
#         print('users not in our DB:', len(notInDb))

# '''CREATES AND STORES PICKLE FILES OF DELTA TIME WISE GRAPHS. EACH GRAPH HAS INFORMATION FOR ALL THE USERS AND THEIR ACTIVITY IN THIS
# SPECIFIC TIME DELTA.'''
# def get_delta_graphs(col):
#     start, end = get_date_of_oldest_newest_document(col)
#     until = start + timedelta(days=PER_DAYS)
#     until = datetime.combine(until.date(), datetime.min.time())

#     while until.date() <= end.date():
#         since = until - timedelta(days=PER_DAYS)
#         GG=pickle.load(open('graphs/weighted_' + str(until.strftime('%m_%d_%Y')), 'rb'))
#         userInfo = {}
#         for node in tqdm(G.nodes()):
#             userInfo[node]={'hashtags':[],'urls':[],'tweets':0 ,'degree': str(G.degree[node]), 'weight': str(int(G.degree(weight='weight')[node])),'label':str(node)}
#         print (userInfo)
#         it = col.find({"created_at": {"$lte": until,"$gte":since}})
#         users=set()
#         for j in it:
#             user=j['user']['id_str']
#             users.add(user)
#         for u in tqdm(list(users)):
#             iterator = col.find({"$and": [{'user.id_str': u}, {"created_at": {"$lte": until, "$gte": since}}]}).sort('created_at', 1)
#             userTags=[]
#             tweets=0
#             userUrls=[]
#             for i in iterator:
#                 tweets += 1
#                 if 'entities' in i:
#                     for h in i['entities']['hashtags']:
#                         foundhashtag = h['text'].lower()
#                         # print (foundhashtag)
#                         userTags.append(foundhashtag)
#                     for n in i['entities']['urls']:
#                         foundurl = (n['expanded_url'])
#                         # print (foundurl)
#                         userUrls.append(foundurl)
#             obj={'hashtags':userTags,'urls':userUrls,'tweets':tweets,'degree':str(G.degree[u]),'weight':str(int(G.degree(weight='weight')[u])),'label':u}
#             userInfo[u] = obj
#         nx.set_node_attributes(G, userInfo)
#         pickle.dump(G,open('graphs/delta_'+ str(until.strftime('%m_%d_%Y')), 'wb'))
#         until = until + timedelta(days=PER_DAYS)

'''CREATES AND STORES PICKLE FILES OF CUMULATIVE GRAPHS. EACH GRAPH HAS INFORMATION FOR ALL THE USERS AND THEIR ACTIVITY UP TO THAT TIME'''
def get_cumulative_graphs(col):
    start, end = get_date_of_oldest_newest_document(col)
    until = start + timedelta(days=PER_DAYS)
    until = datetime.combine(until.date(), datetime.min.time())
    bot_score = pickle.load(open('bot_score', 'rb'))
    while until.date() <= end.date():
        since = until - timedelta(days=PER_DAYS)
        GG=pickle.load(open('graphs/weighted_' + str(until.strftime('%m_%d_%Y')), 'rb'))
        G = nx.subgraph(GG, bot_score.keys())
        userInfo = {}
        for node in G.nodes():
            userInfo[node] = {'hashtags':[],'urls':[],'tweets':0,'retweets':0,'degree':str(G.degree[node]),'weight':str(int(G.degree(weight='weight')[node])),'label':node,'friends':0,'followers':0}
        print (userInfo)
        it = col.find({"created_at": {"$lte": until,"$gte":start}})
        users=set()
        for j in tqdm(it):
            user=j['user']['id_str']
            if user in bot_score.keys():
                users.add(user)
        for u in tqdm(list(users)):
            iterator = col.find({"$and": [{'user.id_str': u}, {"created_at": {"$lte": until, "$gte": start}}]}).sort('created_at', 1)
            userTags=[]
            tweets=0
            retweets=0
            userUrls=[]
            for i in iterator:
                tweets += 1
                friends= i['user']['friends_count']
                fols = i['user']['followers_count']
                if 'retweeted_status' in i:
                    retweets+=1
                if 'entities' in i:
                    for h in i['entities']['hashtags']:
                        foundhashtag = h['text'].lower()
                        # print (foundhashtag)
                        userTags.append(foundhashtag)
                    for n in i['entities']['urls']:
                        foundurl = (n['expanded_url'])
                        # print (foundurl)
                        userUrls.append(foundurl)
            obj={'hashtags':userTags,'urls':userUrls,'tweets':tweets,'retweets':retweets,'degree':str(G.degree[u]),'weight':str(int(G.degree(weight='weight')[u])), 'label':u,'friends':friends,'followers':fols}
            userInfo[u] = obj
        nx.set_node_attributes(G, userInfo)
        pickle.dump(G,open('graphs/cumulative_'+ str(until.strftime('%m_%d_%Y')), 'wb'))
        until = until + timedelta(days=PER_DAYS)
