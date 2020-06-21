ARG REGION=us-east-1

# SageMaker PyTorch image
FROM 520713654638.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:1.5.1-cpu-py3
# FROM 520713654638.dkr.ecr.$REGION.amazonaws.com/sagemaker-pytorch:0.4.0-cpu-py3
# FROM 763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:1.5.1-cpu-py36-ubuntu16.04
ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /src /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# this environment variable is used by the SageMaker PyTorch container to determine our program entry point
# for training and serving.
# For more information: https://github.com/aws/sagemaker-pytorch-container
ENV SAGEMAKER_PROGRAM train.py
