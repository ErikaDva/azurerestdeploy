#%% packages
# run once !pip install flask
# !set FLASK_APP=app.py
from flask import Flask, request, jsonify
import json
import requests
from collections import OrderedDict
from model_class import MultiClassNet
import torch
#%% local development
# local_file_path = 'model_iris_state.pt'
# model.load_state_dict(torch.load(local_file_path))

#%% Model from Google Cloud Storage
# download the model state dict
URL = 'https://storage.googleapis.com/deploy_iris_model/model_iris_state.pt'
r = requests.get(URL)
local_file_path = 'downloaded_model.pt'
with open(local_file_path, 'wb') as f:
    f.write(r.content)
    f.close()
    
#%% load the downloaded model
model = MultiClassNet(HIDDEN_FEATURES=6, NUM_CLASSES=3, NUM_FEATURES=4)
model.load_state_dict(torch.load(local_file_path))


#%%
app = Flask(__name__)
@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'GET':
        return 'Please use POST method and pass data'
    
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        print(content_type)
        if (content_type == 'application/json'):
            data = request.data.decode('utf-8')
            
            print(request.data)
            dict_data = json.loads(data.replace("'", "\""))
            print(dict_data['data'])
            X = torch.tensor([dict_data['data']])
            
            y_test_hat_softmax = model(X)
            y_test_hat = torch.max(y_test_hat_softmax.data, 1)
            y_test_cls = y_test_hat.indices.cpu().detach().numpy()[0]
            y_test_cls

            cls_dict = {
                0: 'setosa', 
                1: 'versicolor', 
                2: 'virginica'
            }

            result = f"Your flower belongs to class {cls_dict[y_test_cls]}."
        return result

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True)
