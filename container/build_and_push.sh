#!/usr/bin/env bash

# This script builds the Docker image and push it to ECR to be ready for use by SageMaker.
# ----------------------------------------------------------------------------------------

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-east-1 if none defined)
region=$(aws configure get region)
region=${region:-us-east-1}

# Define the new ECR image name
image="sagemaker-tennis"
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.
aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1
if [ $? -ne 0 ]
then
    aws ecr create-repository --repository-name "${image}" > /dev/null
fi

# Get the login command from ECR and execute it directly
$(aws ecr get-login --region ${region} --no-include-email) 2> /dev/null

# Get the login command from ECR in order to pull down the SageMaker PyTorch image
$(aws ecr get-login --registry-ids 520713654638 --region ${region} --no-include-email) 2> /dev/null

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
cd $HOME/SageMaker/sagemaker-tennis/container
docker build -t ${image} . --build-arg REGION=${region}
docker tag ${image} ${fullname}

docker push ${fullname}
cd $HOME/SageMaker/sagemaker-tennis
