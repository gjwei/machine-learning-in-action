#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/11
  
"""
import feedparser
import re


def get_word_counts(url):
    d = feedparser.parse(url)
    wc = {}
    
    for e in d.entries:
        if 'summary' in e:
            summary = e.summary
        else:
            summary = e.description
        
        # Extract a list of words
        words = getwords(e.title + ' ' + summary)
        for word in words:
            wc.setdefault(word, 0)
            wc[word] += 1
    return d.feed.title, wc


def getwords(html):
    # 去除html所有标记
    txt = re.compile(r'<[^>]+>').sub('', html)
    
    # Split words by all non-alpha characters
    words = re.compile(r'[^A-Z^a-z]+').split(txt)
    
    # Convert to lowercase
    return [word.lower() for word in words if word != '']


apcount = {}
wordcounts = {}
feedlist = [line for line in file('feedlist.txt')]
for feedurl in feedlist:
    try:
        feedurl = feedurl.strip()

        print("get word from ", feedurl)
        title, wc = get_word_counts(feedurl)
        wordcounts[title] = wc
        for word, count in wc.items():
            apcount.setdefault(word, 0)
            if count > 1:
                apcount[word] += 1
    except:
        print 'Failed to parse feed %s' % feedurl

wordlist = []
for w, bc in apcount.items():
    frac = float(bc) / len(feedlist)
    if frac > 0.1 and frac < 0.5:
        wordlist.append(w)

out = file('blogdata.txt', 'w')
out.write('Blog')
for word in wordlist:
    out.write('\t%s' % word)
out.write('\n')
for blog, wc in wordcounts.items():
    print blog
    out.write(blog)
    for word in wordlist:
        if word in wc:
            out.write('\t%d' % wc[word])
        else:
            out.write('\t0')
    out.write('\n')
