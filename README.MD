# RE:Agent Prototype Setup Instructions

## Overview
This prototype demonstrates an AI-powered banking compliance system similar to RE:Agent, featuring:
- **RAG-based document analysis** using LlamaIndex
- **Real-time transaction compliance review**
- **Multi-modal data integration** (SQL database + document knowledge base)
- **Interactive Streamlit interface**
- **Intelligent risk assessment** and routing

## Prerequisites
- Python 3.8+
- OpenAI API key
- Basic understanding of banking compliance concepts

## Installation

### 1. Create Virtual Environment
```bash
python -m venv re_agent_env
source re_agent_env/bin/activate  # On Windows: re_agent_env\Scripts\activate
```

### 2. Install Dependencies
Create `requirements.txt`:

```txt
streamlit==1.29.0
llama-index==0.9.15
llama-index-llms-openai==0.1.5
llama-index-embeddings-openai==0.1.6
openai==1.3.7
pandas==2.1.4
sqlite3
python-dotenv==1.0.0
```

Install packages:
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Project Structure
```
re_agent_prototype/
├── re_agent_prototype.py      # Main application
├── requirements.txt           # Dependencies
├── .env                      # Environment variables
├── bank_ai_demo.db           # SQLite database (auto-generated)
└── README.md                 # This file
```

## Running the Prototype

### 1. Start the Application
```bash
streamlit run re_agent_prototype.py
```

### 2. Access the Interface
Open your browser to: `http://localhost:8501`

## System Components

### 1. **Document Knowledge Base**
The system includes three AI-generated compliance documents:
- **AML Policy**: Anti-money laundering rules and thresholds
- **Wire Transfer Manual**: International transfer requirements
- **Regulatory Updates**: Latest BSA/AML compliance requirements

### 2. **Transaction Database**
SQLite database with sample data:
- **Transactions Table**: Transaction details, amounts, risk scores
- **Customers Table**: Customer information, KYC status, PEP flags

### 3. **AI Analysis Engine**
- **LlamaIndex RAG**: Retrieves relevant compliance information
- **OpenAI GPT-4**: Analyzes transactions against regulations
- **Risk Assessment**: Multi-factor risk scoring algorithm
- **Decision Engine**: Determines review requirements

### 4. **Sample Transactions**
- **TXN001**: $25K Business Payment (Low Risk)
- **TXN002**: $95K Real Estate Purchase (Medium Risk)
- **TXN003**: $500K International Trade (High Risk - PEP Customer)
- **TXN004**: $8.5K ATM Withdrawal (Low Risk)

## Key Features Demonstrated

### 1. **Intelligent Document Retrieval**
- RAG system queries compliance documents based on transaction context
- Semantic search finds relevant regulations automatically
- Multi-document integration provides comprehensive guidance

### 2. **Risk-Based Decision Making**
- Automated risk scoring using multiple factors
- Business rules engine determines review requirements
- Explainable AI provides reasoning for decisions

### 3. **Real-Time Compliance Checking**
- Instant analysis of transactions against regulations
- Identification of policy violations and compliance gaps
- Automated generation of recommendations and next steps

### 4. **Audit Trail and Documentation**
- Complete transaction analysis history
- Regulatory reference tracking
- Confidence scoring for AI decisions

## Technical Architecture

```
User Input (Transaction ID)
    ↓
Database Query (Transaction + Customer Data)
    ↓
RAG System (Compliance Document Retrieval)
    ↓
LLM Analysis (GPT-4 Reasoning)
    ↓
Risk Assessment Engine
    ↓
Structured Compliance Result
```

## Extending the Prototype

### Adding New Documents
```python
# Add new compliance documents
new_doc = Document(
    text="Your compliance policy text here...",
    metadata={"source": "filename.pdf", "type": "policy"}
)

# Rebuild index with new documents
documents.append(new_doc)
vector_index = VectorStoreIndex.from_documents(documents)
```

### Adding New Transaction Types
```python
# Extend transaction analysis logic
def analyze_crypto_transaction(self, tx_data):
    # Add cryptocurrency-specific compliance checks
    pass
```

### Enhanced Risk Scoring
```python
# Add machine learning risk models
from sklearn.ensemble import RandomForestClassifier

def ml_risk_score(self, tx_data):
    # Use trained ML model for risk scoring
    return model.predict_proba(features)[0][1]
```

## Demo Script for Interviews

### 1. **System Overview** (2 minutes)
"This prototype demonstrates an AI-powered banking compliance system that transforms manual compliance checking into intelligent, automated guidance."

### 2. **Architecture Walkthrough** (3 minutes)
- Show document knowledge base setup
- Explain RAG retrieval mechanism
- Demonstrate transaction data integration
- Highlight LLM reasoning process

### 3. **Live Transaction Analysis** (5 minutes)
- Start with low-risk transaction (TXN001)
- Progress to high-risk PEP transaction (TXN003)
- Show how system identifies compliance issues
- Demonstrate recommendation generation

### 4. **Technical Deep Dive** (5 minutes)
- Code walkthrough of key components
- Explain vector indexing and semantic search
- Show business rules integration
- Discuss scalability considerations

## Key Talking Points for Interviews

### 1. **Problem Understanding**
"I built this to demonstrate how AI can transform compliance from a manual bottleneck into an intelligent accelerator for banking operations."

### 2. **Technical Innovation**
"The system combines structured data queries with unstructured document analysis, using ReAct-style agents to orchestrate between different knowledge sources."

### 3. **Business Value**
"This approach reduces compliance review time from hours to seconds while improving consistency and reducing regulatory risk."

### 4. **Production Readiness**
"The architecture is designed for enterprise deployment with proper error handling, audit trails, and scalable vector storage."

## Troubleshooting

### Common Issues
1. **OpenAI API Key**: Ensure your API key is valid and has sufficient credits
2. **Dependencies**: Use exact versions specified in requirements.txt
3. **Database**: The SQLite database is created automatically on first run
4. **Streamlit**: If port 8501 is busy, Streamlit will suggest an alternative

### Performance Optimization
- **Vector Index Caching**: Store index to disk for faster startup
- **Batch Processing**: Process multiple transactions simultaneously
- **Model Optimization**: Use smaller embedding models for faster retrieval

This prototype effectively demonstrates the core concepts of RE:Agent while being simple enough to explain and extend during technical interviews.
