# Deployment Layer – Production-Ready Infrastructure

This folder contains all components required to deploy the project into a *production environment. It ensures that the machine learning models and pipelines are **accessible, scalable, and maintainable*.

## Components

### 1. API (FastAPI)
- Provides REST endpoints for model inference.
- Main endpoint: /predict  
  Accepts input data (CSV/JSON) → Returns Forecast results.
- Includes input validation, logging, and error handling.
- Enables integration with external systems (e.g., ERP, web apps).

### 2. Web Application (Streamlit)
- Interactive dashboard for end-users.
- Upload CSV files to visualize predictions and performance metrics.
- Provides real-time insights and simple visualizations for business decision-making.

### 3. Containerization (Docker)
- Ensures consistent environment across machines and servers.
- Dockerfile defines all dependencies for the API and Web App.
- Ready for deployment on any platform supporting Docker.

### 4. CI/CD (GitHub Actions)
- Automates deployment whenever changes are pushed to the repository.
- Pipeline ensures *code quality, model updates, and seamless delivery*.
- Supports integration with testing scripts and monitoring.

## Usage
1. Build Docker containers:
   ```bash
   docker build -t project-api ./deployment/api
   docker build -t project-web ./deployment/web_app
