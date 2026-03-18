# End-to-End ML Analytics System

## Overview
This project is a production-oriented machine learning analytics system for customer churn prediction.
It covers the full ML workflow: data loading, preprocessing, feature engineering, model training, evaluation, and API-based inference.

## Project Goal
The goal is to demonstrate an end-to-end ML engineering workflow that is reproducible, modular, and deployable.

## Tech Stack
- Python
- pandas
- scikit-learn
- FastAPI
- Docker

## Problem Statement
Customer churn directly impacts retention cost and revenue stability.
This project predicts churn risk using customer-level behavioral and service data.

## System Architecture
Raw Data  
→ Validation  
→ Preprocessing  
→ Feature Engineering  
→ Model Training  
→ Evaluation  
→ API Inference

## Repository Structure
- `src/data`: data loading, validation, preprocessing
- `src/features`: feature engineering
- `src/models`: model training, evaluation, prediction
- `src/api`: FastAPI inference service
- `src/pipelines`: train/inference pipelines
- `tests`: unit tests
- `artifacts`: saved models, metrics, figures

## Dataset
Planned dataset: IBM Telco Customer Churn dataset

## Current Status
- [x] Repository initialized
- [ ] Folder structure setup
- [ ] Baseline EDA
- [ ] Baseline model
- [ ] FastAPI inference
- [ ] Dockerization

## Future Work
- Add model comparison
- Add API validation schema
- Add Dockerized deployment
- Add performance reporting