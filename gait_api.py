#!/usr/bin/env python
# coding: utf-8

# ###  Gait API to predict ADLs on WISDM dataset

# In[1]:


## Segmenting samples (windowing) functions 
import scipy.stats as stats

Fs = 20
segment_size =  128# Fs*4 # window size=n_time steps=frame size
step_size =  25 # Fs*2  # step size 
n_features = 3     

def get_segments_test(df_scaled, segment_size, step_size):
    segments = []  
    for i in range(0, len(df_scaled) - segment_size - 1, step_size):
        xs = df_scaled['x_axis'].values[i: i + segment_size]
        ys = df_scaled['y_axis'].values[i: i + segment_size]
        zs = df_scaled['z_axis'].values[i: i + segment_size]
        segments.append(np.dstack([xs, ys, zs]))
    # Bring the segments into a better shape
    segments = np.asarray(segments).reshape(-1, segment_size, n_features)
          
    return segments


# In[2]:


### Feature Extraction function 
import scipy.stats as stats
from scipy.stats import skew,kurtosis
from IPython.display import clear_output
def feature_extraction(df):
    mean=[]
    std=[]
    var=[]
    minn=[]
    maxx=[]
    median=[]
    skew=[]
    kurtos=[]
    for i in df:
        mn=np.mean(i,axis=0) # calculate mean to each window, for each axis on each sensor 
        st=np.std(i,axis=0)  # calculate standard deviation to each window, for each axis on each sensor
        vr=np.var(i,axis=0)  # calculate variance to each window, for each axis on each sensor 
        mi=np.min(i,axis=0) 
        mx=np.max(i,axis=0) 
        med=np.median(i,axis=0) 
        sk=stats.skew(i, axis=0, bias=True)
        kur=stats.kurtosis(i, axis=0, bias=True)
        
        mean.append(mn) 
        std.append(st)
        var.append(vr) 
        minn.append(mi)
        maxx.append(mx)
        median.append(med)
        skew.append(sk)
        kurtos.append(kur)
        
    dfmean=pd.DataFrame(mean)
    dfstd=pd.DataFrame(std)
    dfvar=pd.DataFrame(var)
    dfmi=pd.DataFrame(minn)
    dfmx=pd.DataFrame(maxx)
    dfmed=pd.DataFrame(median)
    dfsk=pd.DataFrame(skew)
    dfkur=pd.DataFrame(kurtos)
    
    dft=pd.concat([dfmean,dfstd,dfvar,dfmi,dfsk,dfmx,dfmed,dfkur],axis=1)
    #dft.columns=
    return dft 


# In[3]:


# define chunking function to split a dict into list of dicts
def split_dict_equally(input_dict, chunks=6):
    "Splits dict by keys. Returns a list of dictionaries."
    # prep with empty dicts
    return_list = [dict() for idx in range(chunks)]
    idx = 0
    for k,v in input_dict.items():
        return_list[idx][k] = v
        if idx < chunks-1:  # indexes start at 0
            idx += 1
        else:
            idx = 0
    return return_list


# ### Flask server

# In[ ]:


from flask import Flask, request,jsonify
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from
#from Model_WISDM import *
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False # keep order of sorted dictionary passed to jsonify() function
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
info = 
    {    
    'title': LazyString(lambda: 'Gait Analysis (GA)'), 
    'version': LazyString(lambda: '1.0.0'),  
    'description': LazyString(lambda: 'This is the documentation of the Gait Analysis (GA) module in ALAMEDA AI Toolkit.'),  
    'Contact':  LazyString(lambda: 'ntnu@email.here')
    },    
    host = LazyString(lambda: request.host)
)

swagger_config = {   
    "headers": [],    
    "specs": [       
        {          
            "endpoint": 'Gait',      
            "route": '/Gait.json',     
            "rule_filter": lambda rule: True,    
            "model_filter": lambda tag: True,       
        }    
    ],  
    "static_url_path": "/flasgger_static", 
    "swagger_ui": True,   
    "specs_route": "/apidocs/"
}
swagger = Swagger(app, template=swagger_template, config=swagger_config)

#@swag_from("gait_ui.yaml", methods=['GET'])
@app.route('/')
def welcome():
    return "Welcome to GA AI Toolkit.\nPlease type localhsot:port/apidocs/ to open in swagger UI"

pickle_in = open("classifier_rf.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/Gait',methods=["POST"])
def Gait():
    """
    Gait class with the highest score.
    Returns the gait with the highest score for a given input CSV file
    ---
    tags:
      - name: ""
        description: "Gait activity estimations for a given CSV file"
    schemes:
      - "https"
    parameters:
      - name: file
        in: formData
        description: "The CSV file to be analyzed"
        type: file
        required: true
      
    responses:
        200:
            description: The output values
                
    """
    df=pd.read_csv(request.files.get("file"))
        
    # Normlaization of data
    scaler = StandardScaler()
    X = df[['x_axis', 'y_axis', 'z_axis']]
    scaler = StandardScaler()
    dx = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(data = dx, columns = X.columns) #scaled data

    X_test = get_segments_test(df_scaled, segment_size, step_size)#Segmentation of data
    X_test_fs=feature_extraction(X_test)    #Feature extraction
    prediction = classifier.predict_proba(X_test_fs)[0] # predict class probability score
    prediction=prediction.round(6)# round of values

    # chunck the dict and rename each label(key) in list of dicts with 'gait_score'  
    # make dictionary with label and probaility score 
    MODEL_LABELS=['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']
    labelprob=split_dict_equally(dict(zip(MODEL_LABELS, prediction)), chunks=6)

    labelprob[0]['gait_score']=labelprob[0].pop('Walking')
    labelprob[1]['gait_score']=labelprob[1].pop('Jogging')
    labelprob[2]['gait_score']=labelprob[2].pop('Upstairs')
    labelprob[3]['gait_score']=labelprob[3].pop('Downstairs')
    labelprob[4]['gait_score']=labelprob[4].pop('Sitting')
    labelprob[5]['gait_score']=labelprob[5].pop('Standing')

    #add key-value (gait_class:label) pair to the dict 'labelprob' 
    list_new=[]
    labels=['Walking','Jogging','Walking_upstairs','Walking_downstairs','Sitting','Standing']
    for i in range (len(labelprob)):
        updict = {"gait_class" : labels[i]}
        res = {**updict, **labelprob[i]}
        list_new.append(res)
    list_new_max = max(list_new, key=lambda x:x['gait_score'])
    print("The gait with the highest score for a given test file")
    
    #insert current date/timestamp
    import datetime
    date=datetime.datetime.utcnow().isoformat() + "Z"

    observations=[{"timestamp": date, 'Gait class with the highest score':list_new_max}]

    # produce final output
    output={"user_id": "abcd1234", "source": "GA",'observations':observations}

    # output in Json format
    return jsonify(output)

@app.route('/Classes',methods=["POST"])
def Classes():
    
    """ 
    ALL gait classes and score 
    Retrieve the scores of each gait class for a given input CSV file and do the following:
    1. Send outputs to SemKG 
    2. Display results in the screen 
    ---
    tags:
      - name: ""
        description: "Gait activity estimations for a given CSV file"
    schemes:
      - "https"
    parameters:
      - name: file
        in: formData
        description: "The CSV file to be analyzed"
        type: file
        required: true
        
      
    responses:
        200:
            description: The output values
         
    """
    df=pd.read_csv(request.files.get("file"))
 
    # Scale data
    scaler = StandardScaler()
    X = df[['x_axis', 'y_axis', 'z_axis']]
    scaler = StandardScaler()
    dx = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(data = dx, columns = X.columns) #scaled data

    X_test = get_segments_test(df_scaled, segment_size, step_size)#Segmentation of data
    X_test_fs=feature_extraction(X_test)     #Feature extraction
    prediction = classifier.predict_proba(X_test_fs)[0] # predict class probability score
    prediction=prediction.round(6)# round of values

    #chunck the dict and rename each label(key) in list of dicts with 'gait_score' and
    #make dictionary with label and probaility score 
    MODEL_LABELS=['Walking','Jogging','Upstairs','Downstairs','Sitting','Standing']
    labelprob=split_dict_equally(dict(zip(MODEL_LABELS, prediction)), chunks=6)

    labelprob[0]['gait_score']=labelprob[0].pop('Walking')
    labelprob[1]['gait_score']=labelprob[1].pop('Jogging')
    labelprob[2]['gait_score']=labelprob[2].pop('Upstairs')
    labelprob[3]['gait_score']=labelprob[3].pop('Downstairs')
    labelprob[4]['gait_score']=labelprob[4].pop('Sitting')
    labelprob[5]['gait_score']=labelprob[5].pop('Standing')

    #add key-value (gait_class:label) pair to the dict 'labelprob' 
    list_new=[]
    labels=['Walking','Jogging','Walking_upstairs','Walking_downstairs','Sitting','Standing']
    for i in range (len(labelprob)):
        updict = {"gait_class" : labels[i]}
        res = {**updict, **labelprob[i]}
        list_new.append(res)
    
    #insert current date/timestamp
    import datetime
    date=datetime.datetime.utcnow().isoformat() + "Z"

    observations=[{"timestamp": date, 'gait_classes':list_new}]
    # produce final output
    output={"user_id": 12345, "source": "GA",'observations':observations}
    
    # send result to api endpoint
    url='http://91.184.203.22:4567/ga/post'
    import requests
    import json
    response= requests.post(url, json.dumps(output))
    print("response from server",response.text)
    print(output)
    # output to screen in Json format
    return jsonify(output)
   
#################### data to be sent to api-endpoint

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)
    #res=Classes() 
    #print(res)


# In[ ]:



# In[ ]:





# In[ ]:




