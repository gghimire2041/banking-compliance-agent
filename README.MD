# RE:Agent Banking Compliance System - Usage Guide

A FastAPI-based ReAct (Reasoning and Acting) AI agent that performs banking compliance analysis with real-time document retrieval using LlamaIndex and Qdrant vector database.

## How the System Works

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │  ReAct Agent    │    │  RAG Document   │
│   Endpoints     │◄──►│  (GPT-4)        │◄──►│  Store          │
│                 │    │                 │    │  (LlamaIndex +  │
│   /analyze/     │    │ 1. Reason       │    │   Qdrant)       │
│   /search/      │    │ 2. Act          │    │                 │
│   /trace/       │    │ 3. Observe      │    │ • BSA Rules     │
│                 │    │ 4. Repeat       │    │ • OFAC Lists    │
└─────────────────┘    └─────────────────┘    │ • AML Policies  │
                                              │ • Wire Transfer │
                                              │ • Bank Policies │
                                              └─────────────────┘
```

### ReAct Agent Process Flow

When you analyze a transaction, the agent follows this iterative process:

1. **Reasoning Step**: GPT-4 analyzes what information is needed next
2. **Action Selection**: Chooses from available actions:
   - `search_documents` - Query compliance knowledge base
   - `check_customer` - Get customer risk profile
   - `calculate_risk` - Compute transaction risk score
   - `check_sanctions` - Screen against OFAC/sanctions lists
   - `validate_transaction` - Apply business rules
   - `final_decision` - Make compliance determination
3. **Observation**: Records results and updates context
4. **Iteration**: Repeats until final decision is reached

### Document Retrieval System

The system uses semantic search to find relevant compliance documents:
- **Vector Embeddings**: Documents are embedded using OpenAI's embedding model
- **Similarity Search**: User queries are matched against document embeddings
- **Source Attribution**: Every result includes original document citations
- **Real-time**: Document lookup happens within the reasoning loop

## Installation & Setup

### Prerequisites

```bash
# Install dependencies
pip install fastapi uvicorn llama-index llama-index-llms-openai llama-index-embeddings-openai llama-index-vector-stores-qdrant qdrant-client openai python-dotenv

# Start Qdrant vector database (Docker required)
docker run -p 6333:6333 qdrant/qdrant

# Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key"
```

### Running the System

```bash
# Start the API server
python fastapi_react_rag_system.py

# The system will:
# 1. Initialize Qdrant vector database
# 2. Create fake regulatory documents
# 3. Index documents for semantic search
# 4. Start FastAPI server on http://localhost:8000
```

## Usage Examples

### 1. Basic Transaction Analysis

Analyze a single transaction for compliance issues:

```bash
curl -X POST "http://localhost:8000/analyze/TXN003"
```

**What happens internally:**
1. Agent retrieves transaction data ($500K wire to UAE)
2. Searches documents for "wire transfer compliance UAE"
3. Finds BSA requirements and OFAC guidance
4. Checks customer profile (discovers PEP status)
5. Screens beneficiary "Global Trading Corp" 
6. Finds sanctions match
7. Makes final decision: BLOCK transaction

**Response structure:**
```json
{
  "transaction_id": "TXN003",
  "analysis_id": "uuid-here",
  "risk_level": "HIGH",
  "requires_review": true,
  "compliance_issues": [
    "SANCTIONS MATCH - Do not process transaction",
    "Wire transfer exceeds review threshold"
  ],
  "recommendations": [
    "Block transaction immediately and report to OFAC",
    "Enhanced due diligence and senior approval required"
  ],
  "regulatory_references": [
    "OFAC Compliance Guidelines",
    "31 CFR 103.33"
  ],
  "confidence_score": 0.9,
  "total_steps": 6,
  "processing_time_ms": 3247
}
```

### 2. View Detailed Reasoning Trace

See exactly how the agent reasoned through the decision:

```bash
# Use the analysis_id from the previous response
curl "http://localhost:8000/analysis/{analysis_id}/trace"
```

**Example reasoning trace:**
```json
[
  {
    "step_number": 1,
    "thought": "This is a large international wire transfer to UAE. I need to search for relevant compliance requirements first.",
    "action_type": "search_documents",
    "action_parameters": {"query": "wire transfer UAE compliance requirements"},
    "observation_success": true,
    "observation_result": {
      "documents_found": 3,
      "relevant_content": [
        {
          "source": "31 CFR 103.33",
          "content": "For funds transfers of $3,000 or more...",
          "topic": "wire_transfers"
        }
      ]
    },
    "sources": ["31 CFR 103.33", "OFAC Compliance Guidelines"]
  },
  {
    "step_number": 2,
    "thought": "Now I need to check the customer profile to understand their risk rating and PEP status.",
    "action_type": "check_customer",
    "action_parameters": {"customer_id": "CUST003"},
    "observation_success": true,
    "observation_result": {
      "name": "Global Trading Corp",
      "is_pep": true,
      "risk_rating": "High"
    }
  }
]
```

### 3. Direct Document Search

Query the compliance knowledge base directly:

```bash
curl "http://localhost:8000/search?query=cash%20reporting%20requirements&top_k=3"
```

**Response shows matching documents:**
```json
[
  {
    "content": "Each bank shall file a report of each deposit, withdrawal, exchange of currency or other payment or transfer...",
    "source": "31 CFR 103.22",
    "score": 0.89,
    "metadata": {
      "type": "regulation",
      "topic": "cash_reporting",
      "authority": "FinCEN"
    }
  }
]
```

### 4. Available Test Transactions

See what sample transactions are available for testing:

```bash
curl "http://localhost:8000/transactions"
```

**Response:**
```json
{
  "transactions": [
    {"id": "TXN001", "description": "$25K Business Payment", "risk": "Low"},
    {"id": "TXN002", "description": "$95K Real Estate Purchase", "risk": "Medium"},
    {"id": "TXN003", "description": "$500K International Trade (PEP)", "risk": "High"},
    {"id": "TXN004", "description": "$15K Cash Deposit", "risk": "Medium"},
    {"id": "TXN005", "description": "$150K Offshore Investment (PEP)", "risk": "High"}
  ]
}
```

### 5. Batch Processing

Analyze multiple transactions in parallel:

```bash
curl -X POST "http://localhost:8000/batch-analyze" \
  -H "Content-Type: application/json" \
  -d '["TXN001", "TXN002", "TXN004"]'
```

**Processing flow:**
- All transactions processed simultaneously using asyncio
- Each gets independent ReAct analysis
- Results returned as array
- Failed analyses return error results rather than failing entire batch

## Document Knowledge Base

The system includes realistic fake documents covering:

### Regulatory Documents
- **Bank Secrecy Act (31 CFR 103.22)**: Cash transaction reporting requirements
- **AML Program Requirements (31 CFR 103.120)**: Customer due diligence rules
- **Wire Transfer Rules (31 CFR 103.33)**: Travel rule and recordkeeping
- **OFAC Sanctions Guidelines**: Screening requirements and penalties

### Internal Bank Policies
- **Transaction Monitoring Thresholds**: Automated triggers and review requirements
- **Risk Rating Matrix**: Customer classification criteria
- **Escalation Procedures**: SAR reporting and approval workflows

### Document Metadata
Each document includes:
- **Source**: Regulatory citation or policy reference
- **Authority**: Issuing agency (FinCEN, OFAC, Bank Compliance)
- **Topic**: Categorization for targeted search
- **Effective Date**: Currency validation
- **Type**: regulation, guidance, or internal_policy

## Real-World Analysis Examples

### Low-Risk Transaction (TXN001)
- $25K business payment to US supplier
- Agent finds: Standard processing acceptable
- Documents consulted: Wire transfer recordkeeping rules
- Decision: APPROVE with standard monitoring

### High-Risk Transaction (TXN003)
- $500K wire to UAE PEP customer
- Agent discovers: 
  - Customer is politically exposed person
  - Beneficiary matches sanctions list
  - Amount exceeds wire transfer review threshold
- Documents consulted: OFAC guidelines, PEP requirements, wire transfer rules
- Decision: BLOCK - sanctions violation

### Medium-Risk Transaction (TXN004)
- $15K cash deposit
- Agent finds: Exceeds BSA cash reporting threshold
- Documents consulted: Bank Secrecy Act cash reporting rules
- Decision: APPROVE with CTR filing requirement

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/analyze/{transaction_id}` | POST | Single transaction analysis |
| `/analysis/{analysis_id}/trace` | GET | Detailed reasoning steps |
| `/search` | GET | Direct document search |
| `/transactions` | GET | Available test transactions |
| `/batch-analyze` | POST | Multiple transaction analysis |

### Response Time Expectations

- **Simple transactions**: 2-4 seconds
- **Complex transactions**: 4-8 seconds
- **Batch processing**: 6-15 seconds depending on count
- **Document search**: <1 second

Processing time depends on:
- Number of ReAct reasoning steps
- Document search complexity  
- OpenAI API response time
- Transaction complexity and risk factors

## Development & Testing

### Adding New Documents

Modify `_create_fake_documents()` in the RAGDocumentStore class to add new compliance documents:

```python
{
    "content": "Your new regulatory text here...",
    "metadata": {
        "source": "12 CFR 21.11",
        "type": "regulation",
        "topic": "new_compliance_area",
        "authority": "OCC"
    }
}
```

### Custom Risk Rules

Modify `_validate_transaction_async()` to add new business rules:

```python
if transaction_type == "ACH" and amount > 25000:
    warnings.append("Large ACH transfer requires additional verification")
```

### Monitoring & Debugging

- **Server logs**: Show ReAct reasoning steps in real-time
- **Analysis traces**: Full audit trail via `/trace` endpoint
- **Document sources**: All retrieved documents attributed in responses
- **Processing metrics**: Response times and step counts included

This system demonstrates how ReAct agents can systematically analyze complex compliance scenarios using real-time document retrieval, providing both accurate decisions and complete explainability for regulatory requirements.