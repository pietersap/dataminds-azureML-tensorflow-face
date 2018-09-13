# 0 Requirements

## 0.1 Azure ML components

1. Experimentation Account: Required for Azure ML Workbench. Contains workspaces, which in turn contain projects. You can add multiple users (seats).
2. Model Management Account: Used to register, maintain and deploy containerized ML services with the CLI: https://docs.microsoft.com/en-gb/azure/machine-learning/desktop-workbench/model-management-cli-reference
3. Azure ML Workbench and CLI

## 0.2 Input files

Input files are read from the azure ML shared folder by the training script. These files are not present in this repository. To reproduce this project, provide the input files in your own shared directory. Required files include:

1. Haarcascade_frontalface_default.xml (for the cascade face detector, included in the opencv-python library)
2. The FACENET model file and weights in Keras format (both .h5 files). When adapting the code slightly, you could also use a .h5 file with contains both the model and the weights. 
3. Training images. Use one subfolder for each person, and name the subfolder like the person.

# 1 Overview

Building face recognition service using the Azure ML workbench and CLI. We are using a pretrained facenet model on which we add a dense layer with softmax activation for classification. We use Keras,a higher-level API on top of Tensorflow. 

The input to the created service is a preprocessed image encoded as a string. 

# 2 Scripts

## 2.1 train.py

Train.py **trains and saves the model**. Images are read from files on the shared directory of the compute target, faces are extracted with a cascade classifier and then these faces are used to train a model. Starting from a pretrained facenet model,we add an extra dense layer with softmax activation to classify our own images into 6 categories:

0. buscemi (Steve Buscemi)
1. clooney (George Clooney)
2. dicaprio (Leonardo Dicaprio)
3. jennifer (Jennifer Aniston)
4. pieter (myself)
5. unknown

Note: you can train with different people by adding more subfolders to the image folder.

The model is trained using the higher-level API Keras with a Tensorflow backend.

Running this script can be done with the Workbench GUI. Select the script (e.g. train.py), add arguments if applicable, and run. The script is then submitted to the compute target (e.g. local). See next section.

We are using the outputs folder and the shared folder. More info:   https://docs.microsoft.com/en-gb/azure/machine-learning/desktop-workbench/how-to-read-write-files

Input training data and the pretrained model are read from the shared folder. This is a requirement, since these files are to large to be saved in the project folder. Otherwise, they would need to be copied to the compute target every time an experiment is submitted. The shared folder is found with the environment variable "AZUREML_NATIVE_SHARE_DIRECTORY" in the workbench env and can be optionally changed in the 'targetname.compute' file of the compute target.

The trained keras model (my_model.h5) is saved to the outputs/ folder. This folder must be named outputs/ and receives special treatment. It is not submitted to the compute target when submitting an experiment. Saving files to the outputs folder is preferable when the script produces file that will change with every experiment (e.g. the resulting model after a run with new settings). They become part of the run history. Any outputs saved to this folder, can be retrieved after a run in the GUI or with the CLI.

## 2.2 score.py

Score.py **generates an API schema (schema.json)** and is also **passed to the CLI to generate a scoring container image**.

Score.py must be run as an "experiment" first. The schema.json file is saved by run.py to the outputs/ folder. When running as an experiment (via the GUI or with *az ml experiment submit*), the code below "if __name__ == '__main__'" will run. This part of the code is not relevant inside the service container image when deploying.

The init() and run(..) methods are required for deploying the service:
1. Init() defines what happens when the service is first started. In our case, this is mainly loading the model.
2. Run(input_bytes) defines what happens with an input request. Since the schema was created with DataTypes.STANDARD, it expects a string as input. This string is an encoded version of an input, which is decoded by run(input_bytes) to a numpy array and served to the model. It returns a JSON string as output (required, must be JSON serializable. Otherwise the created webservice will not work).

## 2.3 callService.py

Used to the **test the service after deploying**. This reads an image from a file, extracts the face*, preprocesses** (encodes) it and then sends it the service. The data that is sent in the request must be a JSON string with a "input_bytes" key, i.e. exactly matching the name of the argument in run(input_bytes).

Unfortunately, the face must be extracted and processed into an encoded string in advance before sending it to the service. Just uploading an image is not possible. 

*In the current project, the face is extracted with a cascade classifier (instead of using some deep learning method such as object detection). Afterwards, only the extracted face is sent to a neural net for classification. Note that this was also done at training time.

**the code for encoding the image is found in the myImageLibrary.py file. Also note the importance of normalizing the image (dividing pixel values by 255) since this was also done at training time!!!

example usage: python callService.py --url (*service url*) --path (*path to local image*) --key (*authorization key, when using cluster environment*)

## 2.4 webcamdetect.py

Detects people on the webcam, using the service URL. 

# 3 Submitting the experiments (training)

First, the experiments train.py and score.py must be run on the compute target. Train.py will **create a trained model**, score.py will **create a schema.json file**. 

**attach computetarget**

In the example below, an Azure Datascience Virtual Machine is used as remote compute target.

az ml computetarget attach remote -a (hostname/ip-address) -n (targetname) -u (username) [-w (password)]

This creates two files, targetname.runconfig and targetname.compute. These contain information about the connection and configuration.   

More info about compute target types and configuration can be found in the [Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/experimentation-service-configuration-reference)

**prepare the environment (installing dependencies etc.)**

- az ml experiment prepare -c (targetname)


**Submit the training experiment (train.py) to the remote target**

- az ml experiment submit -c (targetname) train.py [--epochs 5]

The script reads files (training data and base model) from the Azure ML shared directory. Make sure these files are present on the target machine in this directory. The shared directory location is set with the environment variable AZUREML_NATIVE_SHARE_DIRECTORY. Save the model files and images in this directory.

View the history of all previous runs with the following command. The run history is stored in the associated storage account and stores output files (most notable the model file) and metrics (if configured) for each run. All files stored in the 'outputs/' folder by the script, are considered outputs to be saved in history.

- az ml history list

**Return generated model**

This command will return the outputs of the experiment (situated on the target computer, in the outputs/ folder of that specific run) back to your local outputs/ folder in the project directory.

- az ml experiment return -r (run-id) -t (target name)

The run id is found in the output of the 'submit' command.

Along with the model, a number_to_text_label.json file is also present in the outputs of the experiment. Copy this file from the outputs/ folder to the root of the working directory. Otherwise, the service will return numbers instead of people's names. If the file remains in outputs/, it will not be present in the service container created in the next section.

# 3 Deploying the service

Our model is now trained and ready to be used. Using this model, we can create a service that takes an image as input and returns the name of the recognized person. 

## 3.1 Setup 

First, an environment must be created. Typically, you will set up a local environment for testing and a cluster environment for deploying. We are talking about deployment environments now, not about compute environments for training! You will also need to create your model management account in advance. Both are one-time steps. If not yet registered, register for the *Microsoft.ContainerRegistry* Resource provider.

The commands below perform these steps. More info can be found [here](https://docs.microsoft.com/en-gb/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-3), under 'prepare to operationalize locally'. The second command creates a *local* deployment environment for testing. Later, we will create a cluster environment in Azure Container Services.

- az provider register --namespace Microsoft.ContainerRegistry 
- az ml env setup -n (new env name) --location "West Europe"
- az ml account modelmanagement create --location "West Europe" -n (new account name) -g (existing resource group name) --sku-name S1

After creating an environment and model management account, do the following: Login in to Azure, set the environment, set the model management account.

- az login
- az ml env set --name (env name) --resource-group (rg name)
- az ml account modelmanagement set --name (account name) --resource-group (rg name)

## 3.2 Deploying locally

Deploying is done in 4 steps: registering the model (with the given model file), creating a manifest (given the score.py file and the schema), building an image from this manifest and creating a service from this image. Thus, the 4 main components in model management are **models, manifests, images and services.**

- az ml model register --model outputs/my_model.h5 --name my_model.h5
- az ml manifest create --manifest-name my_manifest -f score.py -r python -i (*model ID returned by previous command*) -s outputs/schema.json -c aml_config\conda_dependencies.yml 
- az ml image create -n imagefacerecognition --manifest-id (*manifest ID returned by previous command*)
- az ml service create realtime --image-id (*Image ID returned by previous command*) -n (service name)

These 4 steps can also be done in one command. More info [here](https://docs.microsoft.com/en-gb/azure/machine-learning/desktop-workbench/tutorial-classifying-iris-part-3).

The registered models, manifests, images and services can be reviewed in the Model Management Account from the Azure portal.

The image is stored in an Azure container registry with a automatically generated name. There is (currently?) no option to store the image in your own registry. Locate the container with "az ml image usage -i (*image ID*)"

After the service was created, you can obtain the URL with the following command. In case of a cluster environment (AKS), this will also return an authorization key: 

- az ml service usage realtime -i (*service-id*)

## 3.3 Deploying in an Azure Container Services (AKS) cluster

AKS is Azure's Kubernetes offering. With just a few commands, we can provision a cluster and deploy our service to it. 

Setup a new **cluster environment** for deploying. 

- az ml env setup -n (*new cluster name*) --location "West Europe" --cluster

Switch to this new environment and set execution context to cluster. The resource group for the -g parameter is created automatically when creating the environment and is typically named *(*new cluster name*)rg*. (also see *az ml env list*).

- az ml env set -n (*new cluster name*) -g (*resource group')
- az ml env cluster

Now, use the *image* created in the previous section and create a service in exactly the same way. 

- az ml service create realtime --image-id (*Image ID*) -n (service name)

Obtain the service URL and authorization key.

- az ml service usage realtime -i (*service-id*)
- az ml service keys realtime -i (*service-id*)

After testing, don't forget to clean up the resources. The AKS cluster environment is expensive, take it down immediately. 



## 3.4 Testing the model

See the **callService.py** script for an example on how to use the service.

- python callService.py --url (*service URL*) --path (*path to test image*) [--key (*authorization key, if applicable. For cluster environments.*)]

The **webcamdetect.py** script reads input from your webcam (if one is present) and outputs the people detected in the frame. Use a local service for this.

- python webcamdetect.py --url (*service URL*) [--key (*if applicable*)]




