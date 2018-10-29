# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.core.model import Model

from keras.models import load_model
import numpy as np
import os
import base64
import json
import glob
import time

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model_path = Model.get_model_path(model_name = "facemodel")
load_model(model_path)

def init():

    time.sleep(20)
    #deploying
    global model
    model_path = Model.get_model_path(model_name = "facemodel")
    model = load_model(model_path)
    # model = load_model("my_model.h5")

def run(input_bytes):

    input_bytes = base64.b64decode(bytes(input_bytes,'utf-8')) 
    img = np.loads(input_bytes)
    prediction = model.predict(x=img)
    index = np.argmax(prediction)
    outDict = {}
    outDict["index"] = str(index)
    outJsonString = json.dumps(outDict)
    return (str(outJsonString))

def generate_api_schema():
    import os
    print("create schema")
    sample_input = "byestring_representing_image"
    inputs = {"input_bytes": SampleDefinition(DataTypes.STANDARD, sample_input)}
    os.makedirs('outputs', exist_ok=True)
    print(generate_schema(inputs=inputs, filepath=os.path.join("outputs","schema.json"), run_func=run))

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':

    # Generate API schema
    generate_api_schema()

    # If you needed the model (large file) in a Workbench run, you would load it from shared directory.
    # Files in the outputs folder are not available in the next run, the outputs folder is attached to its own run.
    # However, in this case we don't need the model file in Workbench runs. The workbench runs are only there to create the schema.json file.

    #When deploying the service...
    # In the CLI, we must pass the model file as an argument when creating the service image. This model file will be available in outputs/ folder after returning it.
    # This model file will be copied into the container image with the same path name. So in the service, you can also load it from outputs/<filename>.h5, see init().


