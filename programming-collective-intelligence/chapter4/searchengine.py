#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/11
  
"""
import urllib2
import re
from bs4 import BeautifulSoup
from urlparse import urljoin
import sqlite3 as sqlite

ignorewords = {'the': 1, 'of': 1, 'to': 1, 'and': 1, 'a': 1, 'in': 1, 'is': 1, 'it': 1}


class Crawler(object):
    def __init__(self, dbname):
        self.con = sqlite.connect(dbname)
    
    def __del__(self):
        self.con.close()
    
    def dbcommit(self):
        self.con.commit()
    
    # 辅助函数，获取条目ID，如果条目不存在，将其加入数据库中
    def get_entry_id(self, table, field, value, createnew=True):
        cur = self.con.execute(
                "select rowid from %s where %s='%s'" % (table, field, value))
        res = cur.fetchone()
        if res == None:
            cur = self.con.execute(
                    "insert into %s (%s) values ('%s')" % (table, field, value))
            return cur.lastrowid
        else:
            return res[0]
        
            # 为每个页面建立索引
    def add_to_index(self, url, soup):
        if self.is_indexed(url):
            return
        print("Index %s" % url)
        
        # 获取单词
        text = self.get_text_only(soup)
        words = self.separate_words(text)
        
        # 得到url的id
        urlid = self.get_entry_id('urllist', 'url', url)
        
        # 将每个单词和该url关联
        for i in range(len(words)):
            word = words[i]
            if word in ignorewords:
                continue
            wordid = self.get_entry_id('wordlist', 'word', word)
            self.con.execute("insert into wordlocation(urlid,wordid,location) values (%d,%d,%d)" % (urlid, wordid, i))
    
    # 从一个HTML网页中提取文字
    def get_text_only(self, soup):
        v = soup.string
        if v == None:
            c = soup.contents
            resulttext = ''
            for t in c:
                subtext = self.gettextonly(t)
                resulttext += subtext + '\n'
            return resulttext
        else:
            return v.strip()
    
    # 根据任何非空白的字符进行处理
    def separate_words(self, text):
        splitter = re.compile('\\W+')
        return [s.lower() for s in splitter.split(text) if s != '']
    
    # 如果url已经建立索引，返回True
    def is_indexed(self, url):
        return False
    
    # 添加一个关联两个网页的连接
    def add_link_ref(self, url_from, url_to, link_text):
        words = self.separateWords(link_text)
        fromid = self.getentryid('urllist', 'url', url_from)
        toid = self.getentryid('urllist', 'url', url_to)
        if fromid == toid: return
        cur = self.con.execute("insert into link(fromid,toid) values (%d,%d)" % (fromid, toid))
        linkid = cur.lastrowid
        for word in words:
            if word in ignorewords: continue
            wordid = self.getentryid('wordlist', 'word', word)
            self.con.execute("insert into linkwords(linkid,wordid) values (%d,%d)" % (linkid, wordid))
    
    # 从一个网页开始进行广度优先搜索
    def crawl(self, pages, depth=2):
        for i in range(depth):
            newpages = {}
            for page in pages:
                try:
                    c = urllib2.urlopen(page)
                except:
                    print "Could not open %s" % page
                    continue
                try:
                    soup = BeautifulSoup(c.read())
                    self.add_to_index(page, soup)
                    
                    links = soup('a')
                    for link in links:
                        if ('href' in dict(link.attrs)):
                            url = urljoin(page, link['href'])
                            if url.find("'") != -1: continue
                            url = url.split('#')[0]  # remove location portion
                            if url[0:4] == 'http' and not self.is_indexed(url):
                                newpages[url] = 1
                            linkText = self.get_text_only(link)
                            self.add_link_ref(page, url, linkText)
                    
                    self.dbcommit()
                except:
                    print "Could not parse page %s" % page
            
            pages = newpages
    
    def create_index_tables(self):
        self.con.execute('CREATE TABLE urllist(url)')
        self.con.execute('CREATE TABLE wordlist(word)')
        self.con.execute('CREATE TABLE wordlocation(urlid,wordid,location)')
        self.con.execute('CREATE TABLE link(fromid INTEGER,toid INTEGER)')
        self.con.execute('CREATE TABLE linkwords(wordid,linkid)')
        self.con.execute('CREATE INDEX wordidx ON wordlist(word)')
        self.con.execute('CREATE INDEX urlidx ON urllist(url)')
        self.con.execute('CREATE INDEX wordurlidx ON wordlocation(wordid)')
        self.con.execute('CREATE INDEX urltoidx ON link(toid)')
        self.con.execute('CREATE INDEX urlfromidx ON link(fromid)')
        self.dbcommit()


if __name__ == '__main__':
    crawler = Crawler('searchindex.db')
    crawler.create_index_tables()
