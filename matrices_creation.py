import pickle
import networkx as nx
import numpy as np
from scipy.sparse import hstack, vstack, csc_matrix, csr_matrix
from tqdm import tqdm
from numpy import linalg
from sparsesvd import sparsesvd

def matrix_calculation(G):
    adj_list = nx.adjacency_matrix(G)
    unique_hashtags = list()
    unique_urls = list()
    for n, a in G.nodes(data=True):
        if 'hashtags' in a:
            for h in a['hashtags']:
                if h not in unique_hashtags:
                    unique_hashtags.append(h)
        if 'urls' in a:
            for u in a['urls']:
                if u not in unique_urls:
                    unique_urls.append(u)
    hash_to_index = dict()
    for ind, hashtag in enumerate(unique_hashtags):
        hash_to_index.update({hashtag: ind})
    url_to_index = dict()
    for ind, url in enumerate(unique_urls):
        url_to_index.update({url: ind})
    hashtag_matrix = csr_matrix((0, len(unique_hashtags)), dtype=int)
    url_matrix = csr_matrix((0, len(unique_urls)), dtype=int)
    for n, a in G.nodes(data=True):
        hashtag_vector = np.zeros(len(unique_hashtags))
        url_vector = np.zeros(len(unique_urls))
        if 'hashtags' in a:
            for hashtag in a['hashtags']:
                hashtag_index = hash_to_index.get(hashtag)
                hashtag_vector[hashtag_index] = 1
        if 'urls' in a:
            for url in a['urls']:
                url_index = url_to_index.get(url)
                url_vector[url_index] = 1
        hashtag_matrix = vstack([hashtag_matrix, hashtag_vector])
        url_matrix = vstack([url_matrix, url_vector])
    pickle.dump(hashtag_matrix, open('user_hashtag_'+G.name, 'wb'))
    pickle.dump(url_matrix, open('user_url_'+G.name, 'wb'))
    return hashtag_matrix,url_matrix,adj_list

def coupled_matrix_using_weights(G,attribute):
    adj_list = nx.adjacency_matrix(G)
    hashtag_matrix = pickle.load(open('user_hashtag_'+G.name, 'rb'))  # scipy sparse matrix
    url_matrix = pickle.load(open('user_url_'+G.name, 'rb'))  # scipy sparse matrix
    # stack matrices horizontally
    if attribute=='hashtags':
        A = hashtag_matrix
        A = hstack([adj_list,hashtag_matrix])
        # A = hstack([hashtag_matrix,adj_list,url_matrix])
    elif attribute=='urls':
        A = url_matrix
        A = hstack([url_matrix, adj_list,hashtag_matrix])
    A = A.todense()
    return A
    u, s, v = linalg.svd(A,full_matrices=False)
    pickle.dump(u,open('u_only_hash_'+G.name,'wb'))
    pickle.dump(v, open('v_only_hash_'+G.name, 'wb'))
    pickle.dump(s, open('s_only_hash_'+G.name, 'wb'))


