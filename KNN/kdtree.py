#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 9/18/17
  
"""
import numpy as np

from collections import namedtuple
from operator import itemgetter
from pprint import pformat


class Node(object):
    """
    KDTree Node
    """
    
    def __init__(self, split_attribute, value, left_child, right_child, parent):
        """
        
        :param split_attribute: 按照那种属性来划分
        :param value: 该节点值
        :param left_child: 左孩子
        :param right_child: 右孩子
        :param parent: 父节点
        """
        self.split_attribute = split_attribute
        self.value = value
        self.left_child = left_child
        self.right_child = right_child
        self.parent = parent
    
    def is_left(self):
        return (not self.is_root()) and self.parent.left_child == self
    
    def is_right(self):
        return (not self.is_root()) and self.parent.right_child == self
    
    def is_root(self):
        return self.parent is None
    
    def neighbor_node(self):
        if self.is_left():
            return self.parent.right_child
        elif self.is_right():
            return self.parent.left_child
        return None


def KD_tree(points, depth=0):
    try:
        k = len(points[0])
    except IndexError as e:
        return None
    axis = depth % k
    
    points = sorted(points, key=lambda x: x[axis])
    median = len(points) // 2
    left_tree = KD_tree(points[:median], depth + 1)
    right_tree = KD_tree(points[median + 1:], depth + 1)
    node = Node(axis, points[median], left_tree, right_tree, Node)
    
    if left_tree is not None:
        left_tree.parent = node
    
    if right_tree is not None:
        right_tree.parent = node
    
    return node


def distance(p1, p2):
    return np.sqrt(np.sum(np.power(p1 - p2, 2)))


def searchKDTree(node, point, k=1):
    if len(node.value) != len(point):
        return None
    axis = node.split_attribute         # 切分的维度
    value = point[axis]                 # 要搜索的点在指定维度的值
    
    nodeT = node                        #
    while nodeT is not None:
        if value <= node.value[axis]:
            node = nodeT
            nodeT = node.left_child
        else:
            node = nodeT
            nodeT = node.right_child
    # back
    curPoint = node
    curDis = distance(curPoint.value, point)
    nodeT = node
    while node is not None and (not node.is_root()):
        if node.neighbor_node() is not None:
            dis = distance(point, node.neighbor_node().value)
            if dis < curDis:
                curPoint = node.neighbor_node()
                curDis = dis
        if not node.is_root():
            dis = distance(point, node.parent.value)
            if dis < curDis:
                curPoint = node.parent
                curDis = dis
        node = node.parent
    return curPoint


def main():
    """Example usage"""
    point_list = np.array([(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)])
    tree = KD_tree(point_list)
    test_point = (4, 7)
    node = searchKDTree(tree, test_point)
    print(node)
    
    
if __name__ == '__main__':
    main()
