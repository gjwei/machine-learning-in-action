#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/24/17
  
"""
import numpy as np
import math
import operator

class Tree(object):
    """Recurisive implement of decision tree"""
    
    def calculate_shannon_ent(self, dataset):
        """计算香农信息熵"""
        number_entries = len(dataset)
        label_counts = {}
        
        for feature_vec in dataset:
            current_label = feature_vec[-1]
            label_counts[current_label] = label_counts.get(current_label, 0) + 1
            
        shannon_entropy = 0
        
        for key in label_counts:
            prob = float(label_counts[key]) / number_entries
            shannon_entropy -= prob * math.log(prob)
            
        return shannon_entropy
    
    def create_dataset(self):
        dataset = [[1, 1, 'yes'],
                   [1, 0, 'no'],
                   [1, 1, 'yes'],
                   [0, 1, 'no'],
                   [0, 1, 'no']]
        label = ['no surfacing', 'flippers']
        return dataset, label
    
    def split_dataset(self, dataset, axis, value):
        return_dataset = []
        for feature_vec in dataset:
            if feature_vec[axis] == value:
                reduce_feature_vector = feature_vec[:axis]
                reduce_feature_vector.extend(feature_vec[axis+1:])
                return_dataset.append(reduce_feature_vector)
        return return_dataset
    
    def choose_best_feature_to_split(self, dataset):
        number_features = len(dataset[0]) - 1
        base_entropy = self.calculate_shannon_ent(dataset)
        base_info_gain = 0
        best_feature = -1
        
        for i in range(number_features):
            feature_list = [example[i] for example in dataset]
            unique_values = set(feature_list)
            new_entropy = 0
            
            for value in unique_values:
                sub_dataset = self.split_dataset(dataset, i, value)
                prop = len(sub_dataset) / float(len(dataset))
                new_entropy += prop * self.calculate_shannon_ent(sub_dataset)
            
            info_gain = base_entropy - new_entropy
            if info_gain > base_info_gain:
                base_info_gain = info_gain
                best_feature = i
            
        return best_feature
    
    def majority_count(self, class_list):
        """找到数据中的class中出现最多的类型"""
        class_count = {}
        
        for vote in class_list:
            class_count[vote] = class_count.get(vote, 0) + 1
        
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1),
                                    reverse=True)
        return sorted_class_count[0][0]
    
    def create_tree(self, dataset, label):
        """创建🌲的函数"""
        class_list = [data[-1] for data in dataset]
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]
        if len(dataset[0]) == 1:
            return self.majority_count(class_list)
        
        best_feature = self.choose_best_feature_to_split(dataset)
        best_feature_label = label[best_feature]
        
        my_tree = {best_feature_label: {}}
        del label[best_feature]
        feature_value = [data[best_feature] for data in dataset]
        unique_values = set(feature_value)
        
        for value in unique_values:
            sub_labels = label[:]
            my_tree[best_feature_label][value] = self.create_tree(self.split_dataset(dataset, best_feature, value),
                                                                  sub_labels)
            
        return my_tree
        
        
if __name__ == '__main__':
    tree = Tree()
    dataset, label = tree.create_dataset()
    print(tree.calculate_shannon_ent(dataset))
    print(dataset)
    print(tree.split_dataset(dataset, 0, 1))
    print(tree.choose_best_feature_to_split(dataset))
    my_tree = tree.create_tree(dataset, label)
    
    print(my_tree)