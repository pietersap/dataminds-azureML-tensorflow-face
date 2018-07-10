# Required Azure ML components

1. Experimentation Account: Required for Azure ML Workbench. Contains workspaces, which in turn contain projects. You can add multiple users (seats).
2. Model Management Account: Used to register, maintain and deploy containerized ML services with the CLI: https://docs.microsoft.com/en-gb/azure/machine-learning/desktop-workbench/model-management-cli-reference

# Overview

Building face recognition service using the Azure ML workbench and CLI. We are using a pretrained facenet model on which we add a dense layer with softmax activation for classification. The input to the service is a preprocessed image encoded as a string. 

# Scripts

## train.py

Train.py **trains and saves the model**. Images are read from file, faces are extracted with a cascade classifier and then these faces are used to train a model. Starting from a pretrained facenet model,we add an extra dense layer with softmax activation to classify our own images into 6 categories:

0. buscemi (Steve Buscemi)
1. clooney (George Clooney)
2. dicaprio (Leonardo Dicaprio)
3. jennifer (Jennifer Aniston)
4. pieter (myself)
5. unknown

The model is trained using the higher-level API Keras with a Tensorflow backend.

Running this script can be done with the Workbench GUI. Select the script (e.g. train.py), add arguments if applicable, and run. The script is then submitted to the compute target (e.g. local). Can also be done with the CLI.

We are using the outputs folder and the shared folder. More info:   https://docs.microsoft.com/en-gb/azure/machine-learning/desktop-workbench/how-to-read-write-files

Input training data and the pretrained model are read from the shared folder. This is a requirement, since these files are to large to be saved in the project folder. Otherwise, they would need to be copied to the compute target every time an experiment is submitted. The shared folder is found with the environment variable "AZUREML_NATIVE_SHARE_DIRECTORY" in the workbench env.

The trained keras model (my_model.h5) is saved to the outputs/ folder. This folder must be named outputs/ and receives special treatment. It is not submitted to the compute target when submitting an experiment. Any outputs saved to this model, can be retrieved after a run in the GUI by selecting the run and saving the outputs to any folder. Files saved to the outputs folder is preferable when the script produces file that will change with every experiment (e.g. the resulting model after a run with new settings). They become part of the run history.

After training, save the model file in the project root directory so that it can be picked up by the score.py file (an alternative is to use the shared folder once again).

## score.py

Score.py **generates an API schema (schema.json)** and is also **passed to the CLI to generate a scoring container image**.

Score.py must be run as an "experiment" first. Make sure the required output files from the train.py script are saved to the project root directory first. The schema.json file is saved by run.py to the outputs/ folder. Retrieve this file and also save it to the working directory.

The init() and run(..) methods are required. 

Init() defines what happens when the service is first started. In our case, this is mainly loading the model.

Run(input_bytes) defines what happens with an input request. Since the schema was created with DataTypes.STANDARD, it expects a string as input. This string is an encoded version of an input, which is decoded by run(input_bytes) to a numpy array and served to the model. It returns a JSON string as output (required, must be JSON serializable. Otherwise the created webservice will not work). 

## callService.py

Used to the **test the service after deploying**. This reads an image from a file, extracts the face*, preprocesses** (encodes) it and then sends it the service. The data that is sent in the request must be a JSON string with a "input_bytes" key, i.e. exactly matching the name of the argument in run(input_bytes).

Unfortunately, the face must be extracted and processed into an encoded string in advance before sending it to the service. Just uploading an image is not possible. 

*In the current project, the face is extracted with a cascade classifier (instead of using some deep learning method such as object detection). Afterwards, only the extracted face is sent to a neural net for classification. Note that this was also done at training time.

**the code for encoding the image is found in the myImageLibrary.py file. Also note the importance of normalizing the image (dividing pixel values by 255) since this was also done at training time!!!

example usage: python callService.py --url (*service url*) --path (*path to local image*) --key (*authorization key, when using cluster environment*)

# Deploying

When the model and schema files (my_model.h5 and schema.json) are in place, the service can be deployed. Run these command in the project root:

## Setup 

First, a compute environment must be created. Typically, you will set up a local environment for testing and a cluster environment for deploying. You will also need your model management account.

More info can be found here, under 'prepare to operationalize locally': https://docs.microsoft.com/en-gb/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-3

These commands do the following: Login in to Azure, set the environment, set the model management account.

1. az login
2. az ml env set --name (env name) --resource-group (rg name)
3. az ml account modelmanagement set --name (account name) --resource-group (rg name)

## Deploying

Deploying is done in 4 steps: registering the model (with the given model file), creating a manifest (given the score.py file and the schema), building an image from this manifest and creating a service from this image. Thus, the 4 main components in model management are **models, manifests, images and services.**

1. az ml model register --model my_model.h5 --name my_model.h5
2. az ml manifest create --manifest-name my_manifest -f score.py -r python -i (*model ID returned by previous command*) -s schema.json -c aml_config\conda_dependencies.yml 
3. az ml image create -n imagefacerecognition --manifest-id (*manifest ID returned by previous command*)
4. az ml service create realtime --image-id (*Image ID returned by previous command*) -n (service name)

These 4 steps can also be done in one command. https://docs.microsoft.com/en-gb/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-3

The registered models, manifests, images and services can be reviewed in the Model Management Account from the Azure portal.

The image is stored in an Azure container registry with a automatically generated name. There is (currently?) no option to store the image in your own registry. Locate the container with "az ml image usage -i (*image ID*)"

After the service was created, you can obtain the URL with the following command. In case of a cluster environment (AKS), this will also return an authorization key: **az ml service usage realtime -i (..)**

See the callService.py script for an example on how to use the service.

See the docs for more info and options.

# Webcamtest and deployToEdge folders (TO DO)

Testing the application in a container so that maybe we can do face recognition on the edge...??





