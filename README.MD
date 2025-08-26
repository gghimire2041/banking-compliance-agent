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

# Windows: Download Docker Desktop and minikube from official websites
# Linux: Install docker, minikube, and kubectl via package manager
```

### Environment Setup

1. **Clone and setup project:**
```bash
git clone [your-repo]
cd compliance-agent
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set OpenAI API key:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Deployment Methods

#### Method 1: Local Development (FastAPI Only)

For quick development and testing:

```bash
# Start Qdrant vector database
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Run the application
python fastapi_react_rag_system.py
```

Access at: http://localhost:8000/docs

#### Method 2: Docker Containerization

For consistent environment across systems:

```bash
# Build container image
docker build -t compliance-api:latest .

# Run with Docker
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-api-key" \
  -e QDRANT_URL="http://qdrant:6333" \
  --link qdrant \
  compliance-api:latest
```

#### Method 3: Kubernetes Deployment (Recommended)

For production-like environment with high availability:

**Step 1: Start Kubernetes Cluster**
```bash
# Option A: Minikube
minikube start

# Option B: Docker Desktop Kubernetes
# Enable Kubernetes in Docker Desktop Settings → Kubernetes → Enable Kubernetes
```

**Step 2: Prepare Application**
```bash
# Ensure Docker is running
docker ps

# Build and load container image
docker build -t compliance-api:latest .

# For minikube only (skip if using Docker Desktop)
minikube image load compliance-api:latest
```

**Step 3: Create Configuration**
Create `k8s-manifests.yaml` with proper structure:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: banking-compliance
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qdrant
  namespace: banking-compliance
spec:
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
      volumes:
      - name: qdrant-storage
        emptyDir: {}
---
apiVersion: v1
kind: Service
metadata:
  name: qdrant-service
  namespace: banking-compliance
spec:
  selector:
    app: qdrant
  ports:
  - port: 6333
    targetPort: 6333
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: compliance-api
  namespace: banking-compliance
spec:
  replicas: 2
  selector:
    matchLabels:
      app: compliance-api
  template:
    metadata:
      labels:
        app: compliance-api
    spec:
      containers:
      - name: api
        image: compliance-api:latest
        imagePullPolicy: Never  # Use local image for minikube
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-api-key
        - name: QDRANT_URL
          value: "http://qdrant-service:6333"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: compliance-api-service
  namespace: banking-compliance
spec:
  selector:
    app: compliance-api
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

**Step 4: Deploy Application**
```bash
# Create secret for OpenAI API key
kubectl create secret generic api-secrets \
  --from-literal=openai-api-key="your-openai-api-key" \
  -n banking-compliance

# Deploy all services
kubectl apply -f k8s-manifests.yaml

# Wait for deployment to complete
kubectl get pods -n banking-compliance --watch

# Access application
kubectl port-forward service/compliance-api-service 8080:8000 -n banking-compliance
```

Access at: http://localhost:8080/docs

## System Management

### Starting the System

**Full startup sequence:**
```bash
# 1. Start Docker
# macOS/Windows: Launch Docker Desktop application
# Linux: sudo systemctl start docker

# 2. Verify Docker is running
docker ps

# 3. Start Kubernetes cluster
minikube start  # or ensure Docker Desktop Kubernetes is enabled

# 4. Deploy application
kubectl apply -f k8s-manifests.yaml

# 5. Check deployment status
kubectl get pods -n banking-compliance

# 6. Access application (when all pods are Running)
kubectl port-forward service/compliance-api-service 8080:8000 -n banking-compliance
```

### Stopping the System

**Graceful shutdown:**
```bash
# Stop port forwarding (Ctrl+C in terminal)

# Stop application pods (keeps configuration)
kubectl scale deployment compliance-api --replicas=0 -n banking-compliance
kubectl scale deployment qdrant --replicas=0 -n banking-compliance

# Stop minikube cluster (preserves all configurations)
minikube stop

# Stop Docker Desktop (optional)
# macOS/Windows: Quit Docker Desktop application
# Linux: sudo systemctl stop docker
```

**Complete cleanup (removes everything):**
```bash
# Delete all application resources
kubectl delete namespace banking-compliance

# Stop and delete minikube cluster
minikube delete

# Remove Docker images (optional)
docker rmi compliance-api:latest
docker rmi qdrant/qdrant:latest
```

### Restarting the System

**Resume after stopping:**
```bash
# 1. Start Docker (if stopped)
# Launch Docker Desktop or: sudo systemctl start docker

# 2. Start minikube
minikube start

# 3. Restore application (configurations still exist)
kubectl apply -f k8s-manifests.yaml

# 4. Wait for pods to be ready
kubectl get pods -n banking-compliance

# 5. Access application
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

### Common Deployment Issues

**Docker not running:**
```bash
# Symptoms: "Cannot connect to the Docker daemon"
# Solution: Start Docker Desktop or run: sudo systemctl start docker
docker ps  # Should show containers or empty list, not error
```

**Image pull errors in Kubernetes:**
```bash
# Symptoms: ErrImagePull or ImagePullBackOff
# Solution: Add imagePullPolicy: Never for local images
# Check that image exists: minikube image ls | grep compliance-api
# Rebuild if needed: docker build -t compliance-api:latest .
# Reload: minikube image load compliance-api:latest
```

**Pod crash loops:**
```bash
# Symptoms: CrashLoopBackOff status
# Diagnosis: kubectl logs deployment/compliance-api -n banking-compliance
# Common causes:
# - Missing environment variables (check secret creation)
# - Incorrect YAML indentation
# - Application startup errors

# Fix environment variables:
kubectl describe secret api-secrets -n banking-compliance
kubectl delete secret api-secrets -n banking-compliance
kubectl create secret generic api-secrets --from-literal=openai-api-key="your-key" -n banking-compliance
```

**Service not accessible:**
```bash
# Check pod status
kubectl get pods -n banking-compliance

# Check service configuration
kubectl get services -n banking-compliance

# Verify port forwarding
kubectl port-forward service/compliance-api-service 8080:8000 -n banking-compliance
```

**Minikube issues:**
```bash
# Minikube won't start
minikube delete  # Nuclear option
minikube start

# Context problems
kubectl config current-context
kubectl config use-context minikube
```

### Performance Issues

**Scaling API pods:**
```bash
kubectl scale deployment compliance-api --replicas=3 -n banking-compliance
```

**Increasing resource limits:**
Edit k8s-manifests.yaml and increase resources:
```yaml
resources:
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

**Check resource usage:**
```bash
kubectl top pods -n banking-compliance
kubectl top nodes
```

## Further Improvements

### Production Readiness Enhancements

**Database Migration:**
- Replace SQLite with PostgreSQL for concurrent access
- Implement connection pooling and transaction management
- Add backup and disaster recovery procedures
- Enable data replication across multiple zones

**Security Hardening:**
- Implement OAuth 2.0/SAML integration with enterprise identity providers
- Add TLS encryption for all service communication
- Enable pod security policies and network segmentation
- Implement comprehensive audit logging and monitoring

**Operational Excellence:**
- Add Prometheus metrics collection and Grafana dashboards
- Implement centralized logging with ELK stack
- Create automated testing and deployment pipelines
- Establish monitoring alerts for system health and SLA breaches

**Integration Capabilities:**
- Connect to real banking systems and core processing platforms
- Integrate with live OFAC, SWIFT, and regulatory data feeds
- Build APIs for compliance case management systems
- Add webhook support for real-time transaction monitoring

**Advanced Features:**
- Implement model versioning and A/B testing capabilities
- Add machine learning model retraining pipelines
- Create custom risk scoring algorithms based on historical data
- Develop real-time dashboard for compliance officers

### Suggested Next Steps

1. **Enterprise Database**: Migrate to PostgreSQL with proper schema design
2. **Authentication System**: Implement enterprise-grade authentication and authorization
3. **Real Data Sources**: Connect to actual banking transaction feeds and regulatory APIs  
4. **Compliance UI**: Build dedicated interfaces for compliance officers and analysts
5. **Advanced Analytics**: Add trending analysis, anomaly detection, and predictive compliance features
6. **Regulatory Certification**: Ensure compliance with banking examination requirements and audit standards

This system provides a solid foundation for AI-powered banking compliance but requires these enhancements for production deployment in regulated financial institutions.