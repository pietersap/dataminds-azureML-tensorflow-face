
import cv2 
import pickle

img_raw = cv2.imread("images/test_images/buscemi_test1.jpg")
img_pkl = pickle.dumps(img_raw) #img_pkl is of type bytes
#img_utf8 = img
#call the webservice here (see also the removeThis sample project in workbench)
#headers = {'Content-Type': 'application/json'}
#if key is not None and key is not []:
#key is only needed when deploying to AKS cluster. Local service does not require key. Default = None.
#headers['Authorization'] = 'Bearer ' + key
#data = '{"input_df": [{"image base64 string": "' + img_pkl + '"}]}'
data = img_pkl
print(data[0:2])
# startTime = time.time()
# res = requests.post(url,headers=headers,data=data)
# try:
#     resDict = json.loads(res.json())
#     apiDuration   = int(float(resDict['executionTimeMs']))
#     localDuration = int(float(1000.0*(time.time() - startTime)))
#     print(resDict)
# except:
#     print("ERROR: webservice returned message " + res.text)

model = load_model("my_model.h5")


def run(input_bytes):
       
    # Predict using appropriate functions
    # prediction = model.predict(input_df)
    img = np.loads(input_bytes)
    prediction = model.predict(x=img)

    return str(prediction.tolist())

r = run(data)
print(r)