#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/11
  
"""
import pydelicious
from deliciouserc import initializeUserDict
from recommendations import get_recommmendations, getRecommendedItems, calculate_similarity_items


def load_movielens(path='./data/movielens'):
    movies = {}
    for line in open(path + '/u.item'):
        (id, title) = line.split('|')[0:2]
        movies[id] = title
        
    prefs = {}
    for line in open(path + '/u.data'):
        (user, movieid, rating, ts) = line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movies[movieid]] = float(rating)
        
    return prefs

if __name__ == '__main__':
    prefs = load_movielens()
    print(prefs['87'])
    
    print(get_recommmendations(prefs, '87'))[:3]
    
    print(getRecommendedItems(prefs, calculate_similarity_items(prefs), '87'))[:5]