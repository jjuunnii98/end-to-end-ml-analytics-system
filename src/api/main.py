from fastapi import FastAPI

app = FastAPI(title="End-to-End ML Analytics System API")


@app.get("/health")
def health_check():
    return {"status": "ok"}