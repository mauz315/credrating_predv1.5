# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:43:18 2019

@author: P900017
"""

# LIME explainer

def explain_tree(data, periods, ratings, model, train_set, sov_lab_encoder, le, feat_key, print_exp):
    
    import pandas as pd
    import numpy as np
    from lime import lime_tabular
#    import webbrowser
    
    X_new = np.array(data.loc[feat_key.index].T)
    if sov_lab_encoder is not None:
        pos_sr = feat_key.index.get_loc(feat_key[feat_key["Key"] == 'SovereignRating'].index[0])
        sob_rating = X_new[:, pos_sr].copy()
        X_new[:, pos_sr] = sov_lab_encoder.transform(X_new[:, pos_sr])
    
    # Predicting to check actual prediction
#    pred_calif = np.array([le.iloc[x == list(le.iloc[:,0]),0].index[0] for x in model.predict(X_new)])
    
    X_new = X_new.astype('float')
    
    # features_names = sum([feature_names_key], [])
    # print(features_names)
    class_names = list(le.index)[0:-1]
    class_names.reverse()
    feature_names = list(feat_key.Key) # Usar .index (nombres muy largos) o usar .Key (Ratio y #)
    # Create the the Lime explainer and the lambda function
    categorical_names = {}
    categorical_names[26] = sov_lab_encoder.classes_
    
    explainer = lime_tabular.LimeTabularExplainer(train_set, mode='classification',
                                                  feature_names=feature_names,
                                                  class_names=class_names,
                                                  categorical_features=[26],
                                                  categorical_names=categorical_names,
                                                  discretize_continuous=True)
    
    predict_fn_rf = lambda x: model.predict_proba(x).astype(float)
    
    # explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names,
    #                                                   class_names=class_names, categorical_features=columns,
    #                                                   categorical_names=feature_names_cat, kernel_width=3)
    # Explaining prediction with Lime
    per = pd.DataFrame(list(data.columns), columns=["Periodo"])
    for period in periods:
        print("Explicación para periodo " + str(per.loc[period].Periodo))
        exp = explainer.explain_instance(X_new[period], model.predict_proba, num_features=5, top_labels=ratings)
        exp.show_in_notebook(show_table=True, show_all=False)
        if print_exp:
            av_lab = exp.available_labels()
            for lab in av_lab:
                print ('Explicación para rating %s' % class_names[lab])
                display ('\n'.join(map(str, exp.as_list(label=lab))))
                print ()

    #print(exp.available_labels())
    exp.save_to_file('explainer/lime_output.html')
    
def output_guide(feat_key, data):
    
    import pandas as pd
    import numpy as np
    
    per = pd.DataFrame(list(data.columns), columns=["Periodo"]) 
    
    dum = []
    for i in list(feat_key.Key)[:-1]:
        dum.append(int(i[5:]))
    dum.append(999)
    
    feat_key["Srt"] = dum
    feat_key = feat_key.sort_values(by=['Srt'])
    feat_key = feat_key[['Key']]
    print ()
    print ("Codificación ratios:")
    display (feat_key)
#    display (feat_key['Key'])
    print ()
    print ("Número de periodo a explicar:")
    display (per)
    
    
    return (None)
    
    