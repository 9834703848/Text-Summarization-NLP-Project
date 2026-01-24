from fastapi import FastAPI, Form, Request
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response, HTMLResponse
from textSummarizer.pipeline.prediction import PredictionPipeline


text:str = "What is Text Summarization?"

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", tags=["home"], response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/train")
async def training():
    return Response("Training is disabled. Using pre-trained model directly.")


@app.post("/predict")
async def predict_route(text: str = Form(...)):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e), "summary": "Error occurred during summarization"}
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)