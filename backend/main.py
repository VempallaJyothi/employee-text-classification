from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from predict import predict_category

app = FastAPI(title="Employee Text Classification API")

# CORS: allow the React app (Vite runs on port 5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=1000)

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text must not be empty")
        return value


class PredictResponse(BaseModel):
    text: str
    category: str
    confidence: float


@app.get("/")
def home():
    return {"message": "Employee Text Classification API is running"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        result = predict_category(request.text)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return {
        "text": request.text,
        "category": result["category"],
        "confidence": result["confidence"],
    }
