# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:09:27 2019

@author: P900017
"""
from IPython.display import HTML
import pandas as pd

def View(df):
    css = """<style>
    table { border-collapse: collapse; border: 3px solid #eee; }
    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
    table thead th { background-color: #eee; color: #000; }
    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
    padding: 3px; font-family: monospace; font-size: 10px }</style>
    """
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + (df.to_html() + css).replace("\n",'\\') + '\';'
    s += '</script>'
    return(HTML(s+css))

def var_cod(feat_key):
    dum = []
    for i in list(feat_key.Key)[:-1]:
        dum.append(int(i[5:]))
    dum.append(999)    
    feat_key["Srt"] = dum
    feat_key = feat_key.sort_values(by=['Srt'])
    feat_key = feat_key[['Key']]
    return(feat_key)

def data_handler(ticker):
    pd.set_option('display.max_columns', 999)
    data = pd.read_csv('data/data_em_1212_0119.csv', sep=',', index_col = ['Ticker'], encoding = "latin1")
    display (data.loc[ticker])