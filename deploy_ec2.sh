#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Configuration
APP_NAME="iris-classifier-app"
AWS_REGION="us-east-1"
INSTANCE_TYPE="t4g.small"
AMI_ID="ami-029f1e8b2d0665554" # Latest Ubuntu 22.04 LTS (ARM64) in us-east-1

echo "===================================================="
echo "      Iris Classifier AWS EC2 Deployment Script"
echo "===================================================="

# 1. Verify AWS CLI
if ! command -v aws &> /dev/null; then
    echo "[ERROR] AWS CLI is not installed. Please install it first."
    exit 1
fi

# 2. Check AWS Credentials
echo "[INFO] Verifying AWS credentials..."
if ! aws sts get-caller-identity &> /dev/null; then
    echo "[ERROR] AWS credentials are invalid or expired."
    exit 1
fi

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
echo "[INFO] AWS Account ID: ${AWS_ACCOUNT_ID}"
echo "[INFO] Target AWS Region: ${AWS_REGION}"

# 3. Create Security Group
echo "[INFO] Configuring Security Group..."
SG_NAME="${APP_NAME}-sg"
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=${SG_NAME}" --query "SecurityGroups[0].GroupId" --output text --region "${AWS_REGION}" 2>/dev/null || echo "")

if [ -z "${SG_ID}" ] || [ "${SG_ID}" == "None" ]; then
    echo "[INFO] Creating Security Group '${SG_NAME}'..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "${SG_NAME}" \
        --description "Security group for Iris Classifier Streamlit dashboard" \
        --region "${AWS_REGION}" \
        --query "GroupId" \
        --output text)
        
    # Authorize port 80 (HTTP)
    aws ec2 authorize-security-group-ingress \
        --group-id "${SG_ID}" \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0 \
        --region "${AWS_REGION}" > /dev/null
        
    # Authorize port 22 (SSH) for debug access (optional, can be restricted)
    aws ec2 authorize-security-group-ingress \
        --group-id "${SG_ID}" \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 \
        --region "${AWS_REGION}" > /dev/null
        
    echo "[INFO] Security Group created: ${SG_ID}"
else
    echo "[INFO] Security Group already exists: ${SG_ID}"
fi

# 4. Configure IAM Instance Profile for ECR Read Access
echo "[INFO] Configuring IAM Instance Profile..."
ROLE_NAME="${APP_NAME}-ec2-role"
PROFILE_NAME="${APP_NAME}-ec2-profile"

ROLE_ARN=$(aws iam get-role --role-name "${ROLE_NAME}" --query "Role.Arn" --output text 2>/dev/null || echo "")

if [ -z "${ROLE_ARN}" ]; then
    echo "[INFO] Creating IAM Role '${ROLE_NAME}'..."
    
    # Create trust policy document
    cat <<EOF > /tmp/ec2-trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF
    
    aws iam create-role \
        --role-name "${ROLE_NAME}" \
        --assume-role-policy-document file:///tmp/ec2-trust-policy.json > /dev/null
        
    # Attach ECR ReadOnly Policy
    aws iam attach-role-policy \
        --role-name "${ROLE_NAME}" \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly > /dev/null
        
    rm -f /tmp/ec2-trust-policy.json
    echo "[INFO] IAM Role created successfully."
else
    echo "[INFO] IAM Role already exists."
fi

# Create Instance Profile if it doesn't exist
PROFILE_ARN=$(aws iam get-instance-profile --instance-profile-name "${PROFILE_NAME}" --query "InstanceProfile.Arn" --output text 2>/dev/null || echo "")

if [ -z "${PROFILE_ARN}" ]; then
    echo "[INFO] Creating Instance Profile '${PROFILE_NAME}'..."
    aws iam create-instance-profile --instance-profile-name "${PROFILE_NAME}" > /dev/null
    aws iam add-role-to-instance-profile --instance-profile-name "${PROFILE_NAME}" --role-name "${ROLE_NAME}" > /dev/null
    echo "[INFO] Instance Profile created and configured."
    # Wait for IAM role configuration to propagate
    sleep 5
else
    echo "[INFO] Instance Profile already exists."
fi

# 5. Generate User Data Script
echo "[INFO] Generating User Data script..."
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPO_URL="${ECR_REGISTRY}/${APP_NAME}"

cat <<EOF > /tmp/ec2-user-data.sh
#!/bin/bash
set -e

# Update and upgrade packages
apt-get update -y

# Install Docker
apt-get install -y docker.io
systemctl start docker
systemctl enable docker

# Install AWS CLI v2 (ARM64 version for Graviton)
apt-get install -y unzip curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Login to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Pull and run the Streamlit container (mapping port 80 to container port 8501)
docker pull ${ECR_REPO_URL}:latest
docker run -d -p 80:8501 --name iris-app --restart always ${ECR_REPO_URL}:latest
EOF

# 6. Launch EC2 Instance
echo "[INFO] Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --security-group-ids "${SG_ID}" \
    --iam-instance-profile Name="${PROFILE_NAME}" \
    --user-data file:///tmp/ec2-user-data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${APP_NAME}-instance}]" \
    --region "${AWS_REGION}" \
    --query "Instances[0].InstanceId" \
    --output text)

echo "[INFO] Launched Instance ID: ${INSTANCE_ID}"
rm -f /tmp/ec2-user-data.sh

# 7. Get Public IP Address
echo "[INFO] Waiting for instance to get a public IP address..."
sleep 5

PUBLIC_IP=""
RETRIES=0
while [ -z "${PUBLIC_IP}" ] || [ "${PUBLIC_IP}" == "None" ]; do
    if [ ${RETRIES} -eq 6 ]; then
        echo "[WARNING] Could not retrieve Public IP automatically. Please check your EC2 Console."
        break
    fi
    sleep 3
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids "${INSTANCE_ID}" \
        --region "${AWS_REGION}" \
        --query "Reservations[0].Instances[0].PublicIpAddress" \
        --output text)
    RETRIES=$((RETRIES+1))
done

echo "===================================================="
echo "              Deployment Successful!"
echo "===================================================="
echo "Instance ID: ${INSTANCE_ID}"
echo "Dashboard URL: http://${PUBLIC_IP}"
echo "Note: It will take 1-2 minutes for the system to boot,"
echo "install Docker, and launch the dashboard container."
echo "===================================================="
