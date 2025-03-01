"""FastAPI application for serving the procurement-granite model."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from procurement_granite.data.document_processor import DocumentProcessor
from procurement_granite.model.inference import GraniteInference, load_inference_model
from procurement_granite.utils.config import load_config, get_project_root


# Define request and response models
class RfpAnalysisRequest(BaseModel):
    """Request model for RFP analysis."""
    
    text: str


class RfpAnalysisResponse(BaseModel):
    """Response model for RFP analysis."""
    
    requirements: str
    evaluation_criteria: str
    technical_specifications: str


class BidEvaluationRequest(BaseModel):
    """Request model for bid evaluation."""
    
    rfp_text: str
    bid_text: str


class BidEvaluationResponse(BaseModel):
    """Response model for bid evaluation."""
    
    score: str
    strengths_weaknesses: str


# Create FastAPI app
app = FastAPI(
    title="Procurement Document Analysis API",
    description="API for analyzing procurement documents and evaluating bids using IBM GRANITE 3.2 8B Instruct.",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
inference_model = None
document_processor = None
config = None


def get_inference_model():
    """Get the inference model.
    
    Returns:
        GraniteInference: The inference model.
    """
    global inference_model
    if inference_model is None:
        inference_model = load_inference_model()
    return inference_model


def get_document_processor():
    """Get the document processor.
    
    Returns:
        DocumentProcessor: The document processor.
    """
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor()
    return document_processor


def get_config():
    """Get the configuration.
    
    Returns:
        dict: The configuration.
    """
    global config
    if config is None:
        config = load_config()
    return config


@app.get("/")
async def root():
    """Root endpoint.
    
    Returns:
        dict: A welcome message.
    """
    return {"message": "Welcome to the Procurement Document Analysis API"}


@app.post("/analyze-rfp", response_model=RfpAnalysisResponse)
async def analyze_rfp(
    request: RfpAnalysisRequest,
    inference: GraniteInference = Depends(get_inference_model),
):
    """Analyze an RFP document.
    
    Args:
        request: The request containing the RFP text.
        inference: The inference model.
        
    Returns:
        RfpAnalysisResponse: The analysis results.
    """
    try:
        # Extract requirements
        requirements = inference.extract_rfp_requirements(request.text)
        
        # Identify evaluation criteria
        evaluation_criteria = inference.identify_evaluation_criteria(request.text)
        
        # Extract technical specifications
        technical_specifications = inference.extract_technical_specifications(request.text)
        
        return RfpAnalysisResponse(
            requirements=requirements,
            evaluation_criteria=evaluation_criteria,
            technical_specifications=technical_specifications,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing RFP: {str(e)}")


@app.post("/evaluate-bid", response_model=BidEvaluationResponse)
async def evaluate_bid(
    request: BidEvaluationRequest,
    inference: GraniteInference = Depends(get_inference_model),
):
    """Evaluate a bid against an RFP.
    
    Args:
        request: The request containing the RFP and bid text.
        inference: The inference model.
        
    Returns:
        BidEvaluationResponse: The evaluation results.
    """
    try:
        # Score bid
        score = inference.score_bid(request.rfp_text, request.bid_text)
        
        # Identify strengths and weaknesses
        strengths_weaknesses = inference.identify_strengths_weaknesses(request.rfp_text, request.bid_text)
        
        return BidEvaluationResponse(
            score=score,
            strengths_weaknesses=strengths_weaknesses,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating bid: {str(e)}")


@app.post("/upload-rfp")
async def upload_rfp(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    processor: DocumentProcessor = Depends(get_document_processor),
    inference: GraniteInference = Depends(get_inference_model),
):
    """Upload and analyze an RFP document.
    
    Args:
        file: The RFP document file.
        background_tasks: Background tasks.
        processor: The document processor.
        inference: The inference model.
        
    Returns:
        dict: The analysis results.
    """
    try:
        # Check file extension
        file_extension = Path(file.filename).suffix.lower().lstrip(".")
        if file_extension not in processor.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format: {file_extension}. Supported formats: {processor.supported_formats}"
            )
        
        # Save file temporarily
        temp_dir = Path(get_project_root()) / "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = temp_dir / file.filename
        
        with open(temp_file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document
        text = processor.process_document(temp_file_path)
        
        # Delete temporary file
        if background_tasks:
            background_tasks.add_task(os.remove, temp_file_path)
        else:
            os.remove(temp_file_path)
        
        # Analyze RFP
        requirements = inference.extract_rfp_requirements(text)
        evaluation_criteria = inference.identify_evaluation_criteria(text)
        technical_specifications = inference.extract_technical_specifications(text)
        
        return {
            "filename": file.filename,
            "requirements": requirements,
            "evaluation_criteria": evaluation_criteria,
            "technical_specifications": technical_specifications,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/upload-bid")
async def upload_bid(
    rfp_file: UploadFile = File(...),
    bid_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    processor: DocumentProcessor = Depends(get_document_processor),
    inference: GraniteInference = Depends(get_inference_model),
):
    """Upload and evaluate a bid against an RFP.
    
    Args:
        rfp_file: The RFP document file.
        bid_file: The bid document file.
        background_tasks: Background tasks.
        processor: The document processor.
        inference: The inference model.
        
    Returns:
        dict: The evaluation results.
    """
    try:
        # Check file extensions
        rfp_extension = Path(rfp_file.filename).suffix.lower().lstrip(".")
        bid_extension = Path(bid_file.filename).suffix.lower().lstrip(".")
        
        if rfp_extension not in processor.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported RFP file format: {rfp_extension}. Supported formats: {processor.supported_formats}"
            )
        
        if bid_extension not in processor.supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported bid file format: {bid_extension}. Supported formats: {processor.supported_formats}"
            )
        
        # Save files temporarily
        temp_dir = Path(get_project_root()) / "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_rfp_path = temp_dir / rfp_file.filename
        temp_bid_path = temp_dir / bid_file.filename
        
        with open(temp_rfp_path, "wb") as f:
            content = await rfp_file.read()
            f.write(content)
        
        with open(temp_bid_path, "wb") as f:
            content = await bid_file.read()
            f.write(content)
        
        # Process documents
        rfp_text = processor.process_document(temp_rfp_path)
        bid_text = processor.process_document(temp_bid_path)
        
        # Delete temporary files
        if background_tasks:
            background_tasks.add_task(os.remove, temp_rfp_path)
            background_tasks.add_task(os.remove, temp_bid_path)
        else:
            os.remove(temp_rfp_path)
            os.remove(temp_bid_path)
        
        # Evaluate bid
        score = inference.score_bid(rfp_text, bid_text)
        strengths_weaknesses = inference.identify_strengths_weaknesses(rfp_text, bid_text)
        
        return {
            "rfp_filename": rfp_file.filename,
            "bid_filename": bid_file.filename,
            "score": score,
            "strengths_weaknesses": strengths_weaknesses,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")


def start_server():
    """Start the FastAPI server."""
    config = get_config()
    uvicorn.run(
        "procurement_granite.api.app:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        workers=config["api"]["workers"],
        timeout_keep_alive=config["api"]["timeout"],
    )


if __name__ == "__main__":
    start_server() 