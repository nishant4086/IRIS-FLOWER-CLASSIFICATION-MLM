#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
APP_NAME="iris-classifier-app"
DEFAULT_REGION="us-east-1"

echo "===================================================="
echo "      Iris Classifier AWS Deployment Script"
echo "===================================================="

# 1. Verify AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "[ERROR] AWS CLI is not installed. Please install it first."
    exit 1
fi

# 2. Check AWS Credentials
echo "[INFO] Verifying AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "[ERROR] AWS credentials are invalid or expired."
    echo "Please run 'aws configure' or set environment variables before running this script."
    exit 1
fi

# Retrieve AWS account info
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
# Force AWS region to us-east-1 because AWS App Runner is not supported in ap-south-1
AWS_REGION="us-east-1"

echo "[INFO] Using AWS Account ID: ${AWS_ACCOUNT_ID}"
echo "[INFO] Using AWS Region: ${AWS_REGION}"

ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO_URL="${ECR_REGISTRY}/${APP_NAME}"

# 3. Create AWS ECR Repository if it doesn't exist
echo "[INFO] Checking ECR repository: ${APP_NAME} in region ${AWS_REGION}..."
if ! aws ecr describe-repositories --repository-names "${APP_NAME}" --region "${AWS_REGION}" &> /dev/null; then
    echo "[INFO] Creating ECR repository '${APP_NAME}' in region '${AWS_REGION}'..."
    aws ecr create-repository --repository-name "${APP_NAME}" --region "${AWS_REGION}" > /dev/null
    echo "[INFO] ECR Repository created successfully."
else
    echo "[INFO] ECR Repository already exists."
fi

# 4. Login to AWS ECR
echo "[INFO] Authenticating Docker with AWS ECR..."
aws ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# 5. Build and Tag Docker Image
echo "[INFO] Building Docker image..."
docker build -t "${APP_NAME}" .

echo "[INFO] Tagging Docker image for ECR..."
docker tag "${APP_NAME}:latest" "${ECR_REPO_URL}:latest"

# 6. Push to AWS ECR
echo "[INFO] Pushing Docker image to AWS ECR..."
docker push "${ECR_REPO_URL}:latest"
echo "[INFO] Image successfully pushed to ECR."

# 7. Check if App Runner service exists, and update/create it
echo "[INFO] Checking AWS App Runner service status in region ${AWS_REGION}..."
SERVICE_ARN=$(aws apprunner list-services --region "${AWS_REGION}" --query "ServiceSummaryList[?ServiceName=='${APP_NAME}-service'].ServiceArn" --output text)

if [ -n "${SERVICE_ARN}" ]; then
    echo "[INFO] App Runner Service exists (ARN: ${SERVICE_ARN}). Triggering deployment..."
    aws apprunner start-deployment --service-arn "${SERVICE_ARN}" --region "${AWS_REGION}" > /dev/null
    echo "[INFO] Deployment triggered. It will take a few minutes to complete."
else
    echo "[INFO] App Runner Service does not exist. Creating a new service..."
    echo "[WARNING] Accessing private ECR requires an AppRunnerECRAccessRole."
    echo "[INFO] Attempting to find/create ECR Access Role ARN..."
    
    # Try to find standard AppRunnerECRAccessRole
    ROLE_ARN=$(aws iam get-role --role-name AppRunnerECRAccessRole --query "Role.Arn" --output text 2>/dev/null || echo "")
    
    if [ -z "${ROLE_ARN}" ]; then
        echo "[WARNING] AppRunnerECRAccessRole not found in IAM. Creating one..."
        
        # Create IAM role policy document
        cat <<EOF > /tmp/trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "build.apprunner.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
        
        ROLE_ARN=$(aws iam create-role --role-name AppRunnerECRAccessRole --assume-role-policy-document file:///tmp/trust-policy.json --query "Role.Arn" --output text)
        aws iam attach-role-policy --role-name AppRunnerECRAccessRole --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
        rm -f /tmp/trust-policy.json
        echo "[INFO] Created AppRunnerECRAccessRole: ${ROLE_ARN}"
        # Give AWS time to propagate IAM role changes
        sleep 5
    else
        echo "[INFO] Found existing ECR Access Role: ${ROLE_ARN}"
    fi

    echo "[INFO] Creating App Runner service '${APP_NAME}-service'..."
    aws apprunner create-service \
        --service-name "${APP_NAME}-service" \
        --source-configuration "{
            \"ImageRepository\": {
                \"ImageIdentifier\": \"${ECR_REPO_URL}:latest\",
                \"ImageConfiguration\": {
                    \"Port\": \"8501\"
                },
                \"ImageRepositoryType\": \"ECR\"
            },
            \"AuthenticationConfiguration\": {
                \"AccessRoleArn\": \"${ROLE_ARN}\"
            },
            \"AutoDeploymentsEnabled\": true
        }" \
        --instance-configuration "{
            \"Cpu\": \"1 vCPU\",
            \"Memory\": \"2 GB\"
        }" --region "${AWS_REGION}" > /dev/null
        
    echo "[INFO] App Runner Service creation initiated. Go to AWS Console to track progress."
fi

echo "===================================================="
echo "                  Deployment Script Done!"
echo "===================================================="
