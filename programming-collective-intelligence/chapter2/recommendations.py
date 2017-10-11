#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/11
  
"""
from math import sqrt
from collections import defaultdict

# A dictionary of movie critics and their ratings of a small set of movies
critics = {
    'Lisa Rose': {
        'Lady in the Water': 2.5,
        'Snakes on a Plane': 3.5,
        'Just My Luck': 3.0,
        'Superman Returns': 3.5,
        'You, Me and Dupree': 2.5,
        'The Night Listener': 3.0,
    },
    'Gene Seymour': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 3.5,
        'Just My Luck': 1.5,
        'Superman Returns': 5.0,
        'The Night Listener': 3.0,
        'You, Me and Dupree': 3.5,
    },
    'Michael Phillips': {
        'Lady in the Water': 2.5,
        'Snakes on a Plane': 3.0,
        'Superman Returns': 3.5,
        'The Night Listener': 4.0,
    },
    'Claudia Puig': {
        'Snakes on a Plane': 3.5,
        'Just My Luck': 3.0,
        'The Night Listener': 4.5,
        'Superman Returns': 4.0,
        'You, Me and Dupree': 2.5,
    },
    'Mick LaSalle': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 4.0,
        'Just My Luck': 2.0,
        'Superman Returns': 3.0,
        'The Night Listener': 3.0,
        'You, Me and Dupree': 2.0,
    },
    'Jack Matthews': {
        'Lady in the Water': 3.0,
        'Snakes on a Plane': 4.0,
        'The Night Listener': 3.0,
        'Superman Returns': 5.0,
        'You, Me and Dupree': 3.5,
    },
    'Toby': {'Snakes on a Plane': 4.5,
             'You, Me and Dupree': 1.0,
             'Superman Returns': 4.0},
}


def sim_distance(prefs, p1, p2):
    """
    返回一个有关p1, p2基于距离的相似度评价
    :param prefs: 用户偏好电影的所有的数据
    :param p1: 用户1
    :param p2: 用户2
    :return: 两个用户基于距离的相似度
    """
    # 得到share_item的列表
    share_item = []
    for item in prefs[p1]:
        if item in prefs[p2]:
            share_item.append(item)
    
    if len(share_item) == 0:
        return 0
    
    sum_of_same = sum([(prefs[p1][item] - prefs[p2][item]) ** 2
                      for item in share_item])
    
    return 1 / (1 + sqrt(sum_of_same))


def sim_pearson(prefs, p1, p2):
    """
    计算皮尔逊相似度值
    :param prefs: 用户爱好的项
    :param p1: 用户1
    :param p2: 用户2
    :return: 皮尔逊相似度值
    """
    share_items = []
    for item in prefs[p1]:
        if item in prefs[p2]:
            share_items.append(item)
    
    # 如果二者没有共同之处，返回1
    if len(share_items) == 0:
        return 1
    
    sum_p1 = sum([prefs[p1][item] for item in share_items])
    sum_p2 = sum([prefs[p2][item] for item in share_items])
    
    sum_p1_p2 = sum([prefs[p1][item] * prefs[p2][item] for item in share_items])
    
    sum_p1_p1 = sum([prefs[p1][item] ** 2 for item in share_items])
    sum_p2_p2 = sum([prefs[p2][item] ** 2 for item in share_items])
    
    numerator = sum_p1_p2 - sum_p1 * sum_p2 / len(share_items)
    denominator = (sum_p1_p1 - (sum_p1) ** 2 / len(share_items)) * (
        sum_p2_p2 - (sum_p2) ** 2 / len(share_items)
    )
    if denominator == 0:
        return 0
    return numerator / sqrt(denominator)

# print(sim_pearson(critics, 'Lisa Rose', 'Gene Seymour'))


def top_mathes(prefs, person, n=5, similarity=sim_pearson):
    scores = [(similarity(prefs, person, p), p)
              for p in prefs if p != person]
    
    scores.sort(reverse=True)
    return scores[:n]

# print(top_mathes(critics, 'Toby', n=3))


def get_recommmendations(prefs, person, similarity=sim_pearson):
    """
    Gets recommendations for a person by using a weighted average
    of every other user's rankings
    """
    totals = {}
    similarity_sums = {}
    
    for other in prefs:
        if other == person:
            continue
            
        sim = similarity(prefs, person, other)
        if sim <= 0:
            continue
        for item in prefs[other]:
            
            # 只对自己没有看过的电影进行评价
            if item not in prefs[person] or prefs[person][item] == 0:
                # 相似度*评价值
                totals[item] = totals.get(item, 0) + prefs[other][item] * sim
                similarity_sums[item] = similarity_sums.get(item, 0) + sim
                
    # 建立一个归一化的列表
    rankings = [(totals[item] / similarity_sums[item], item) for item in totals.keys()]
    # print(rankings)
    # 返回已经排好序的列表
    rankings = sorted(rankings, reverse=True, key=lambda x: x[0])
    return rankings
    

# print(get_recommmendations(critics, 'Toby'))


def transform_prefs(prefs):
    result = defaultdict(dict)
    for preson in prefs:
        for item in prefs[preson]:
            result[item][preson] = prefs[preson][item]
    
    return result


# print transform_prefs(critics)


def calculate_similarity_items(prefs, n=10):
    """建立字典，给出这些物品最为相似的其他物品"""
    result = {}
    
    item_prefs = transform_prefs(prefs)
    c = 0
    
    for item in item_prefs.keys():
        c += 1
        if c % 100 == 0:
            print("%d / %d" % (c, len(item_prefs)))
            
        scores = top_mathes(item_prefs, item, n, similarity=sim_distance)
        result[item] = scores
            
    return result
    
# print(calculate_similarity_items(critics))

def getRecommendedItems(prefs, itemMatch, user):
    """
    根据用户现有的打分情况，计算出那些物品和用户喜欢的物品相似，并推荐
    给用户
    
    """
    userRatings = prefs[user]
    scores = {}
    totalSim = {}
    # Loop over items rated by this user
    for (item, rating) in userRatings.items():
        # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:
            # Ignore if this user has already rated this item
            if item2 in userRatings:
                continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # Sum of all the similarities
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity
    # Divide each total score by total weighting to get an average
    rankings = [(score / totalSim[item], item) for (item, score) in
                scores.items()]
    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


itemsim = calculate_similarity_items(critics)
print(getRecommendedItems(critics, itemsim, 'Toby'))