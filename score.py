# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.assets import get_local_path

from keras.models import load_model
import numpy as np
import os

#SHARED_FOLDER = os.environ["AZUREML_NATIVE_SHARE_DIRECTORY"]
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
    # Get the path to the model asset
    # local_path = get_local_path('mymodel.model.link')
    
    # Load model using appropriate library and function
    global model
    print("Loading model from shared folder...")
    model = load_model("my_model.h5")
    

def run(input_bytes):
       
    # Predict using appropriate functions
    # prediction = model.predict(input_df)
    img = np.loads(input_bytes)
    prediction = model.predict(x=img)

    return str(prediction.tolist())

def generate_api_schema():
    import os
    print("create schema")
    sample_input = "byestring_representing_img"
    inputs = {"input_bytes": SampleDefinition(DataTypes.STANDARD, sample_input)}
    os.makedirs('outputs', exist_ok=True)
    print(generate_schema(inputs=inputs, filepath=os.path.join("outputs","schema.json"), run_func=run))

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger

    logger = get_azureml_logger()

    generate_api_schema()

    init()

