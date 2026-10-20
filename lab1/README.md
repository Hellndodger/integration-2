# Speech Commands Lab Project

This project implements a Dockerized ML pipeline for the Google Speech Commands dataset, with a limited subset of classes: yes, no, up, down.

## Project structure

- app/config.py — global constants and artifact paths
- app/dataset.py — data loading and spectrogram conversion
- app/model.py — CNN model definition and artifact saving/loading
- app/train.py — training script and evaluation pipeline
- app/api.py — FastAPI REST API for inference
- app/streamlit_app.py — optional UI client for the API
- tests/test_api.py — integration tests covering the API
- Dockerfile — container image for training and inference
- docker-compose.yml — orchestration for training, API and testing

## Run the project in Docker

1. Build the image and start the training service:
   docker compose run --rm training
2. Start the API service:
   docker compose up --build api
3. Run the automated integration tests:
   docker compose run --rm test

## Inference endpoint

The API exposes the following route:

- POST /predict
  - Accepts a .wav file upload
  - Returns: label, confidence, latency_ms, dataset, model_name

## Required metrics

The final report includes:
- dataset and modality
- model architecture
- memory size of the trained model
- accuracy on the test set
- average inference latency

## Notes

All data processing, model training, and inference are designed to run inside isolated Docker containers as required by the lab methodology.
