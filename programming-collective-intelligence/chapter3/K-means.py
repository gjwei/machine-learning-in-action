#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/11
  
"""
import random
from clusters import pearson

def k_cluster(data, distance=pearson, k = 4):
    ranges = [(min([row[i] for row in data]), max([row[i] for row in data]))
              for i in range(len(data[0]))]
    clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
                 for i in range(len(data[0]))] for j in range(k)]
    
    last_matches = None
    
    for t in range(100):
        print("Iteration %d" % (t))
        best_mathches = [[] for _ in range(k)]
        
        for j in range(len(data)):
            row = data[j]
            best_mathch = 0
            best_distance = distance(clusters[0], row)
            for i in range(k):
                d = distance(clusters[i], row)
                
                if d < best_distance:
                    best_distance = d
                    best_mathch = i
                    
            best_mathches[best_mathch].append(j)
        
        if best_mathches == last_matches:
            break
        last_matches = best_mathches
        
        # 把中心位置移到所欲成员平均位置
        for i in range(k):
            avgs = [0.0] * len(data[0])
            for rowid in best_mathches[i]:
                for m in range(len(data[rowid])):
                    avgs[m] += data[rowid][m]
            
            for j in range(len(avgs)):
                avgs[j] /= len(best_mathches[i])
            clusters[i] = avgs
    return best_mathches
    
