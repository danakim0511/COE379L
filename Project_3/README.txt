Project 3: Hurricane Harvey Damage Classification

Overview:
This project uses neural networks to classify satellite images of buildings affected by Hurricane Harvey as either "damaged" or "not damaged". The project includes data preprocessing, training multiple models, selecting the best-performing model, deploying it as an inference server, and packaging it with Docker.

Folder Structure:
- app.py               -> Inference server script using FastAPI
- model/best_model.h5  -> Trained Keras model file
- requirements.txt     -> List of Python dependencies
- Dockerfile           -> Docker image definition
- docker-compose.yml   -> Configuration to build and run the container
- README.txt           -> This file

Setup Instructions:

1. Local Setup (without Docker):
   a. Install dependencies:
      pip install -r requirements.txt

   b. Run the server:
      uvicorn app:app --host 0.0.0.0 --port 8000

2. Docker Setup:
   a. Build the Docker image:
      docker-compose build

   b. Run the inference server:
      docker-compose up

3. Access the API:
   - GET /summary
     Example
