# Advanced Banking Compliance System

A production-ready ReAct (Reasoning and Acting) AI agent for banking compliance analysis, featuring real-time document retrieval, vector search, and Kubernetes deployment.

## System Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│    FastAPI Server   │    │   ReAct AI Agent    │    │   RAG Document      │
│                     │    │                     │    │   Store             │
│  • REST Endpoints   │◄──►│  • GPT-4 Reasoning  │◄──►│                     │
│  • Async Processing │    │  • Action Execution │    │  • LlamaIndex       │
│  • Batch Analysis   │    │  • Context Memory   │    │  • Qdrant Vector DB │
│  • Error Handling   │    │  • Decision Logic   │    │  • Semantic Search  │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Core Features

### ReAct Agent Intelligence
- **Iterative Reasoning**: Agent thinks through compliance requirements step-by-step
- **Dynamic Actions**: Searches documents, checks customers, validates rules based on context
- **Audit Trail**: Complete trace of reasoning, actions, and observations
- **Explainable Decisions**: Every compliance determination includes supporting rationale

### Document Retrieval System  
- **Vector Search**: Semantic similarity matching using OpenAI embeddings
- **Regulatory Knowledge**: Bank Secrecy Act, OFAC guidance, AML requirements, wire transfer rules
- **Source Attribution**: All retrieved documents include citations and metadata
- **Real-time Updates**: Document search integrated into agent reasoning loop

### Enterprise API Design
- **Async Processing**: Concurrent transaction analysis with FastAPI
- **Batch Operations**: Parallel processing of multiple transactions
- **Error Resilience**: Graceful handling of API failures and system errors
- **Performance Monitoring**: Response times and processing metrics included

## Installation Guide

### Prerequisites

Install required tools:
```bash
# macOS with Homebrew
brew install docker minikube kubectl

# Start Docker Desktop and enable Kubernetes (alternative to minikube)
```

### Environment Setup

1. **Clone and setup project:**
```bash
git clone [your-repo]
cd compliance-agent
python -m venv .venv
source .venv/bin/activate
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set OpenAI API key:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Local Development (Docker Compose)

For quick local development without Kubernetes:

```bash
# Start Qdrant vector database
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Run the application
python fastapi_react_rag_system.py
```

Access at: http://localhost:8000/docs

### Kubernetes Deployment

#### Option 1: Using Minikube

```bash
# Start minikube cluster
minikube start

# Build and load application image
docker build -t compliance-api:latest .
minikube image load compliance-api:latest

# Create OpenAI API key secret
kubectl create secret generic api-secrets \
  --from-literal=openai-api-key="$OPENAI_API_KEY" \
  -n banking-compliance

# Deploy all services
kubectl apply -f k8s/

# Wait for deployment
kubectl wait --for=condition=ready pod -l app=compliance-api -n banking-compliance --timeout=300s

# Access application
kubectl port-forward service/compliance-api-service 8080:8000 -n banking-compliance
```

#### Option 2: Using Docker Desktop Kubernetes

```bash
# Enable Kubernetes in Docker Desktop Settings
# Then deploy directly
kubectl apply -f k8s/
kubectl port-forward service/compliance-api-service 8080:8000 -n banking-compliance
```

## Usage Guide

### API Endpoints

| Endpoint | Method | Purpose | Example |
|----------|--------|---------|---------|
| `/` | GET | Health check | `curl http://localhost:8000/` |
| `/analyze/{transaction_id}` | POST | Single transaction analysis | `curl -X POST http://localhost:8000/analyze/TXN003` |
| `/analysis/{analysis_id}/trace` | GET | Detailed reasoning trace | `curl http://localhost:8000/analysis/{id}/trace` |
| `/search` | GET | Direct document search | `curl "http://localhost:8000/search?query=wire%20transfer"` |
| `/transactions` | GET | Available test transactions | `curl http://localhost:8000/transactions` |
| `/batch-analyze` | POST | Multiple transaction analysis | `curl -X POST -H "Content-Type: application/json" -d '["TXN001","TXN002"]' http://localhost:8000/batch-analyze` |

### Test Scenarios

#### Low-Risk Transaction
```bash
curl -X POST "http://localhost:8000/analyze/TXN001"
```
**Expected Result**: $25K business payment approved with standard monitoring

#### High-Risk Compliance Violation
```bash
curl -X POST "http://localhost:8000/analyze/TXN003"
```
**Expected Result**: $500K UAE PEP transaction blocked due to sanctions match

#### Cash Reporting Requirement
```bash
curl -X POST "http://localhost:8000/analyze/TXN004"
```
**Expected Result**: $15K cash deposit approved with CTR filing requirement

### Viewing Analysis Details

After any analysis, use the returned `analysis_id` to see the reasoning trace:

```bash
# Copy analysis_id from previous response
curl "http://localhost:8000/analysis/35a98ffb-c51e-4d90-8989-8a3d7fe4b39d/trace"
```

This shows the complete ReAct process:
- Step-by-step thoughts and decisions
- Documents retrieved and consulted
- Risk calculations and rule validations
- Source citations for all compliance determinations

### Document Search Testing

Query the regulatory knowledge base directly:
```bash
curl "http://localhost:8000/search?query=cash%20reporting%20requirements&top_k=3"
curl "http://localhost:8000/search?query=OFAC%20sanctions%20screening"
curl "http://localhost:8000/search?query=PEP%20customer%20due%20diligence"
```

## System Monitoring

### Kubernetes Cluster Status
```bash
# Check pod health
kubectl get pods -n banking-compliance

# View application logs
kubectl logs -f deployment/compliance-api -n banking-compliance

# Monitor resource usage
kubectl top pods -n banking-compliance

# Check service status
kubectl get services -n banking-compliance
```

### Performance Metrics
- **Response Times**: 1-8 seconds depending on complexity
- **Throughput**: 10-50 concurrent requests (depending on resources)
- **Memory Usage**: ~256MB per API pod
- **Document Search**: Sub-second semantic similarity queries

## Compliance Knowledge Base

### Regulatory Documents Included
- **Bank Secrecy Act (31 CFR 103.22)**: Cash transaction reporting thresholds
- **AML Program Requirements (31 CFR 103.120)**: Customer due diligence procedures
- **Wire Transfer Compliance (31 CFR 103.33)**: Travel rule and recordkeeping
- **OFAC Sanctions Guidelines**: Screening requirements and violation penalties
- **Customer Due Diligence Rule (31 CFR 1020.220)**: Enhanced due diligence standards
- **Internal Bank Policies**: Transaction monitoring and escalation procedures

### Search Capabilities
- **Semantic Search**: Natural language queries across all documents
- **Metadata Filtering**: Search by document type, authority, or topic
- **Citation Tracking**: All responses include source document references
- **Real-time Retrieval**: Document lookup integrated into agent reasoning

## Development Workflow

### Making Changes

1. **Code Changes**: Modify `fastapi_react_rag_system.py`
2. **Rebuild Image**: `docker build -t compliance-api:latest .`
3. **Load to Minikube**: `minikube image load compliance-api:latest`
4. **Redeploy**: `kubectl rollout restart deployment/compliance-api -n banking-compliance`

### Adding New Documents

Modify `_create_fake_documents()` method:
```python
{
    "content": "New regulatory guidance text...",
    "metadata": {
        "source": "12 CFR 225.4",
        "type": "regulation",
        "topic": "credit_risk",
        "authority": "Federal Reserve"
    }
}
```

### Customizing Risk Rules

Update validation logic in `_validate_transaction_async()`:
```python
if amount > 1000000:  # $1M threshold
    violations.append("Transaction exceeds bank daily limit")
```

## Production Considerations

### Current Limitations
- **Database**: SQLite not suitable for concurrent banking operations
- **Security**: Missing authentication, encryption, and access controls
- **Scale**: Limited to moderate transaction volumes
- **Integration**: No connection to real banking systems or data sources

### Production Enhancements Required
- **Database**: PostgreSQL with connection pooling
- **Authentication**: OAuth 2.0/SAML integration with banking systems
- **Monitoring**: Prometheus metrics and alerting
- **Security**: TLS encryption, secrets management, audit logging
- **Performance**: Redis caching, load balancing, auto-scaling

### Regulatory Compliance Gaps
- **Audit Trails**: Need immutable, tamper-proof logging
- **Data Retention**: Compliance with banking record-keeping requirements
- **Disaster Recovery**: Backup and restore procedures
- **Change Management**: Formal approval process for model updates

## Troubleshooting

### Common Issues

**API not starting:**
```bash
# Check if OpenAI API key is set
echo $OPENAI_API_KEY

# Verify Qdrant is running
kubectl get pods -n banking-compliance
```

**Document search not working:**
```bash
# Check Qdrant service connectivity
kubectl port-forward service/qdrant-service 6333:6333 -n banking-compliance
curl http://localhost:6333/collections
```

**Pod crashes or restarts:**
```bash
# Check pod logs
kubectl logs deployment/compliance-api -n banking-compliance

# Check resource limits
kubectl describe pod -l app=compliance-api -n banking-compliance
```

### Performance Tuning

**Scaling API pods:**
```bash
kubectl scale deployment compliance-api --replicas=3 -n banking-compliance
```

**Increasing resource limits:**
```yaml
# In api-deployment.yaml
resources:
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

## Next Steps

This system demonstrates the core concepts for AI-powered banking compliance but requires significant enhancements for production use:

1. **Database Migration**: Replace SQLite with enterprise database
2. **Security Implementation**: Add authentication and encryption
3. **Real Data Integration**: Connect to actual banking systems and regulatory feeds
4. **UI Development**: Build compliance officer interfaces
5. **Regulatory Certification**: Ensure compliance with banking examination requirements

The current implementation provides a solid foundation for understanding how ReAct agents can systematically analyze complex compliance scenarios using real-time document retrieval and explainable decision making.