# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.assets import get_local_path

from keras.models import load_model
import numpy as np
import os
import base64
import json
import glob


#this fails when creating the service. This environment variable and the shared directory are not available
#in the service container. The shared folder is still useful in the train.py script though, for storing the data,
#the pretrained model and weights,...
#just pick up the trained model manually and store in project directory and use load_model("my_model.h5").

# TO DO
# - add model and weights files
# - set paths to these files in code below
# - add myImageLibrary

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.


def init():

    #deploying
    global model
    print("Loading model...")
    print("GLOB: ",glob.glob('./*'))
    model = load_model("outputs/my_model.h5")
    print("loading model went wrong")


def run(input_bytes):

    input_bytes = base64.b64decode(bytes(input_bytes,'utf-8'))
    #input_bytes2 = bytes(input_bytes,'utf-8')   
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

    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger
    logger = get_azureml_logger()

    # Generate API schema
    generate_api_schema()

    # If you needed the model (large file) in a Workbench run, you would load it from shared directory.
    # Files in the outputs folder are not available in the next run, the outputs folder is attached to its own run.
    # However, in this case we don't need the model file in Workbench runs. The workbench runs are only there to create the schema.json file.

    #When deploying the service...
    # In the CLI, we must pass the model file as an argument when creating the service image. This model file will be available in outputs/ folder after returning it.
    # This model file will be copied into the container image with the same path name. So in the service, you can also load it from outputs/<filename>.h5, see init().

    # print("loading model from shared directory...")
    # SHARED_FOLDER = os.environ["AZUREML_NATIVE_SHARE_DIRECTORY"]
    # model = load_model(os.path.join(SHARED_FOLDER,"my_model.h5"))


