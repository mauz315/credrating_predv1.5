# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:09:27 2019

@author: P900017
"""

#def View(df):
#    css = """<style>
#    table { border-collapse: collapse; border: 3px solid #eee; }
#    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
#    table thead th { background-color: #eee; color: #000; }
#    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
#    padding: 3px; font-family: monospace; font-size: 10px }</style>
#    """
#    s  = '<script type="text/Javascript">'
#    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
#    s += 'win.document.body.innerHTML = \'' + (df.to_html() + css).replace("\n",'\\') + '\';'
#    s += '</script>'
#    return(HTML(s+css))
#
#def var_cod(feat_key):
#    dum = []
#    for i in list(feat_key.Key)[:-1]:
#        dum.append(int(i[5:]))
#    dum.append(999)    
#    feat_key["Srt"] = dum
#    feat_key = feat_key.sort_values(by=['Srt'])
#    feat_key = feat_key[['Key']]
#    return(feat_key)
    
def issuer_choice(data_file):
    
    from ipywidgets import widgets
    from IPython.display import display, clear_output
    import pandas as pd

#    def myfunc(issuer):
#        return data.loc[issuer]['ticker']
    def on_button_clicked(b):
        ticker = data.loc[w.value]['ticker']
        with output:
            clear_output()
            print('Ticker CIQ: ' + ticker)
        clear_output()
        display(w)        
        display(button, output) #, output2)
#        print('Ticker CIQ: ' + ticker)
        data_handler(data_file,ticker)
                    
    data = pd.read_csv('data/ticker_list.csv', sep=',', index_col = 1, encoding = "latin-1")
    w = widgets.Dropdown(options=list(data.index), description='Emisor:')
        
    button = widgets.Button(description="Obtener CIQ")
    output = widgets.Output()
#    w.observe(on_button_clicked, names='value')
    
    display(w)        
    display(button, output) #, output2)
    button.on_click(on_button_clicked)
    
def data_handler(data_file, ticker):
    
    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_csv('data/'+ data_file, sep=',', index_col = ['Ticker'], encoding = "latin1")
    feat_key = pd.read_csv('data/features.csv', sep=',', index_col = ["Key"], encoding = "latin1")
    # pd.set_option('display.max_columns', 999)
    # display (data.loc[ticker])
    cols = data.columns[1:-2]
    l = int(len(cols)/2)
    
    for i in range(0,l):
        ratios = data[[cols[i], cols[i+13], 'Fecha']].loc[ticker]
        ratios['Fecha'] = pd.to_datetime(ratios['Fecha'])
        fig, ax = plt.subplots()
        print(feat_key.loc[cols[i]].Feature)
        ratios.plot(x='Fecha', ax=ax, cmap='Paired', legend=False)
        plt.show()
#        ax.legend([feat_key.loc[cols[i]].Feature,feat_key.loc[cols[i+13]].Feature])
        
        
        