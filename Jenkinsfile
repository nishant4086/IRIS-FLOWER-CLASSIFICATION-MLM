pipeline {
    agent any

    environment {
        DOCKER_REGISTRY = 'docker.io'
        DOCKER_IMAGE = 'nishant4086/iris-classification-mlm'
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        REGISTRY_CREDENTIALS_ID = 'docker-hub-credentials'
    }

    options {
        timeout(time: 1, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
    }

    stages {
        stage('Setup & Install') {
            steps {
                echo 'Setting up Python Virtual Environment...'
                sh '''
                    python3 -m venv venv
                    ./venv/bin/pip install --upgrade pip
                    ./venv/bin/pip install -r requirements.txt
                '''
            }
        }

        stage('Lint & Static Analysis') {
            steps {
                echo 'Running Ruff for linting and code quality checks...'
                sh './venv/bin/ruff check .'
            }
        }

        stage('Train & Validate Model') {
            steps {
                echo 'Running training pipeline to validate ML model flow...'
                sh './venv/bin/python iris_classification.py --train'
            }
            post {
                success {
                    echo 'Archiving training outputs and evaluation visualizations...'
                    archiveArtifacts artifacts: 'outputs/*.png', followSymlinks: false
                }
            }
        }

        stage('Docker Build') {
            steps {
                echo "Building Docker Image: ${DOCKER_IMAGE}:${IMAGE_TAG}"
                sh "docker build -t ${DOCKER_IMAGE}:${IMAGE_TAG} ."
                sh "docker tag ${DOCKER_IMAGE}:${IMAGE_TAG} ${DOCKER_IMAGE}:latest"
            }
        }

        stage('Docker Security Scan') {
            steps {
                echo 'Running vulnerability scan on Docker image...'
                sh '''
                    if command -v trivy >/dev/null 2>&1; then
                        trivy image --severity HIGH,CRITICAL --exit-code 0 ${DOCKER_IMAGE}:${IMAGE_TAG}
                    else
                        echo "[WARNING] Trivy is not installed on the agent. Skipping security scan."
                    fi
                '''
            }
        }

        stage('Docker Push') {
            when {
                branch 'main'
            }
            steps {
                echo 'Pushing Docker image to registry...'
                // Note: Requires Docker credentials configured in Jenkins with ID matching REGISTRY_CREDENTIALS_ID
                /*
                withCredentials([usernamePassword(credentialsId: REGISTRY_CREDENTIALS_ID, usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh 'echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin'
                    sh "docker push ${DOCKER_IMAGE}:${IMAGE_TAG}"
                    sh "docker push ${DOCKER_IMAGE}:latest"
                }
                */
                echo 'Docker push is commented out by default. Configure registry credentials to enable it.'
            }
        }
    }

    post {
        always {
            echo 'Cleaning up workspace...'
            cleanWs()
        }
        success {
            echo 'CI/CD Pipeline finished successfully!'
        }
        failure {
            echo 'CI/CD Pipeline failed. Please check build logs.'
        }
    }
}
