# Procurement Document Analysis with IBM GRANITE

A specialized LLM system for analyzing procurement documents (RFPs) and evaluating bids for connectivity services using fine-tuned IBM Granite 3.1 2B Instruct.

## Project Overview

This project aims to develop an AI system that can:
- Extract key information from Request for Proposal (RFP) documents
- Identify evaluation criteria and technical specifications
- Score bids against RFP requirements
- Provide objective, consistent bid evaluations with justifications

## Technology Stack

- **Base Model**: IBM Granite 3.1 2B Instruct
- **Fine-tuning**: Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- **Framework**: PyTorch, Hugging Face Transformers
- **API**: FastAPI
- **Deployment**: Docker, Cloud Infrastructure with GPU support
- **Evaluation**: RAGAS, Custom procurement metrics

## Project Structure

```
procurement-granite/
├── config/                 # Configuration files
├── data/                   # Data directory
│   ├── raw/                # Raw procurement documents
│   ├── processed/          # Processed text and datasets
│   └── synthetic/          # Synthetic examples
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks for exploration and analysis
├── src/                    # Source code
│   ├── api/                # FastAPI application
│   ├── data/               # Data processing scripts
│   ├── evaluation/         # Evaluation metrics and benchmarks
│   ├── model/              # Model training and inference
│   └── utils/              # Utility functions
├── tests/                  # Test cases
├── .gitignore              # Git ignore file
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
└── README.md               # Project README
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/AI-Unisphere/procurement-llm.git
   cd procurement-granite
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Initialize DVC for data version control:
   ```
   dvc init
   ```

## Development Workflow

1. **Data Collection and Preparation**:
   - Gather procurement documents
   - Process documents into text
   - Create instruction-tuning datasets

2. **Model Development**:
   - Fine-tune IBM GRANITE 3.2 8B Instruct with LoRA
   - Evaluate model performance
   - Optimize for inference

3. **API Development**:
   - Create FastAPI endpoints
   - Implement document processing pipeline
   - Develop structured output schemas

4. **Deployment**:
   - Containerize with Docker
   - Deploy to cloud infrastructure
   - Set up monitoring and feedback collection

## License

[Specify your license here]

## Contributors

[List contributors here] 
