pipeline {
    agent {
        label 'local'
    }
    stages {
        stage('Checkout Code') {
            steps {
                git branch: 'main', url: "https://github.com/AnshBhavsar1654/MLOps-Assignment"
            }
        }
        stage('Show workspace') {
            steps {
                bat 'cd'
            }
        }
        stage('Build Images') {
            steps {
                bat 'docker-compose build'
            }
        }
        stage('Run Containers') {
            steps {
                bat 'docker-compose up -d'
            }
        }
    }
}
