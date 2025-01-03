pipeline {
    agent any
    environment {
        DAGSHUB_MLFLOW_URL = "https://dagshub.com/chirag-35/dags_mlops_jenkins.mlflow"
        MLFLOW_TRACKING_USERNAME = "chirag-35 "
        MLFLOW_TRACKING_PASSWORD = credentials('')
    }
    stages {
        stage('Clone Repository') {
            steps {
                // Clone Repository
                script {
                    echo 'Cloning DagsHub Repository...'
                    
                }
            }
        }
        stage('Install Packages') {
            steps {
                // Install Packages
                script {
                    echo 'Install Python Packages...'
                    sh '''
                        python --version
                        pip install --break-system-packages -r requirements.txt >> req-output.txt
                    '''
                }
            }
        }
        stage('Train Model') {
            steps {
                // Train ML Model
                script {
                    echo 'Training ML Model ...'
                    sh 'python train.py'
                }
            }
        }
    }
}