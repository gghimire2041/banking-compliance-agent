# FastAPI ReAct RAG Banking Compliance System
# Implements proper document retrieval with LlamaIndex, Qdrant vector DB, and async ReAct agents

import os
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from contextlib import asynccontextmanager

# FastAPI and async dependencies
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# LlamaIndex and vector DB
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# Qdrant client
import qdrant_client
from qdrant_client.models import Distance, VectorParams

# OpenAI for ReAct reasoning
import openai

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Configure LlamaIndex
Settings.llm = OpenAI(model="gpt-5-mini", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

class ActionType(Enum):
    SEARCH_DOCUMENTS = "search_documents"
    CHECK_CUSTOMER = "check_customer"
    CALCULATE_RISK = "calculate_risk"
    CHECK_SANCTIONS = "check_sanctions"
    VALIDATE_TRANSACTION = "validate_transaction"
    GET_HISTORICAL = "get_historical"
    FINAL_DECISION = "final_decision"

@dataclass
class AgentAction:
    action_type: ActionType
    parameters: Dict[str, Any]
    reasoning: str

@dataclass
class AgentObservation:
    result: Any
    success: bool
    error_message: Optional[str] = None
    sources: Optional[List[str]] = None

@dataclass
class ReActStep:
    step_number: int
    thought: str
    action: AgentAction
    observation: AgentObservation
    timestamp: datetime

# Pydantic models for API
class TransactionData(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float
    transaction_type: str
    description: str
    country: str
    beneficiary_name: str
    purpose_code: str

class ComplianceResult(BaseModel):
    transaction_id: str
    analysis_id: str
    risk_level: str
    requires_review: bool
    compliance_issues: List[str]
    recommendations: List[str]
    regulatory_references: List[str]
    confidence_score: float
    analysis_summary: str
    total_steps: int
    processing_time_ms: int

class ReActTrace(BaseModel):
    step_number: int
    thought: str
    action_type: str
    action_parameters: Dict[str, Any]
    observation_success: bool
    observation_result: Any
    sources: Optional[List[str]] = None
    timestamp: str

class DocumentSearchResult(BaseModel):
    content: str
    source: str
    score: float
    metadata: Dict[str, Any]

class RAGDocumentStore:
    """Document retrieval system using LlamaIndex and Qdrant"""
    
    def __init__(self):
        self.client = None
        self.index = None
        self.query_engine = None
        self.collection_name = "banking_compliance_docs"
        
    async def initialize(self):
        """Initialize Qdrant client and vector store"""
        try:
            self.client = qdrant_client.QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
            
            # Create collection if it doesn't exist
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
                )
            except Exception:
                pass  # Collection might already exist
            
            # Set up vector store and index
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name
            )
            
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create or load index
            try:
                self.index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                )
            except Exception:
                # If no documents exist, create empty index
                documents = await self._create_fake_documents()
                self.index = VectorStoreIndex.from_documents(
                    documents=documents,
                    storage_context=storage_context
                )
            
            # Set up query engine
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=5
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever
            )
            
            print("Document store initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize document store: {e}")
            # Fallback to in-memory index
            documents = await self._create_fake_documents()
            self.index = VectorStoreIndex.from_documents(documents)
            self.query_engine = self.index.as_query_engine()
    
    async def _create_fake_documents(self) -> List[Document]:
        """Create fake regulatory documents for prototype"""
        
        fake_documents = [
            {
                "content": """
                BANK SECRECY ACT - CASH TRANSACTION REPORTING
                
                Section 103.22 - Reports of cash transactions
                
                (a) Each bank shall file a report of each deposit, withdrawal, exchange of currency or other payment or transfer, by, through, or to such bank which involves a transaction in currency of more than $10,000.
                
                (b) Currency transaction reports must be filed within 15 days of the transaction.
                
                (c) Structured transactions - Multiple currency transactions totaling more than $10,000 in a single day by or for the same person shall be treated as a single transaction if the bank knows the transactions are by or for the same person.
                
                Penalties: Civil penalties up to $75,000 per violation. Criminal penalties up to $250,000 and 5 years imprisonment.
                
                Effective Date: Current as of 2025
                """,
                "metadata": {
                    "source": "31 CFR 103.22",
                    "type": "regulation",
                    "topic": "cash_reporting",
                    "effective_date": "2025-01-01",
                    "authority": "FinCEN"
                }
            },
            {
                "content": """
                ANTI-MONEY LAUNDERING PROGRAM REQUIREMENTS
                
                Section 103.120 - Anti-money laundering programs for banks
                
                (a) Program required. Each bank shall develop and implement a written anti-money laundering program reasonably designed to prevent the bank from being used to facilitate money laundering and the financing of terrorist activities.
                
                (b) Minimum requirements. The program shall, at a minimum:
                (1) Establish policies, procedures, and controls to prevent money laundering
                (2) Designate a compliance officer responsible for assuring day-to-day compliance
                (3) Provide for ongoing employee training programs
                (4) Provide for independent testing for compliance
                
                (c) Customer due diligence. Banks must implement risk-based customer due diligence policies and procedures including:
                - Customer identification and verification
                - Beneficial ownership identification for legal entity customers
                - Understanding the nature and purpose of customer relationships
                - Ongoing monitoring for reporting suspicious transactions
                
                Enhanced due diligence is required for higher-risk customers including:
                - Politically exposed persons (PEPs)
                - Customers from high-risk geographic locations
                - Customers engaged in high-risk activities
                """,
                "metadata": {
                    "source": "31 CFR 103.120",
                    "type": "regulation", 
                    "topic": "aml_program",
                    "effective_date": "2025-01-01",
                    "authority": "FinCEN"
                }
            },
            {
                "content": """
                OFFICE OF FOREIGN ASSETS CONTROL (OFAC) COMPLIANCE
                
                Sanctions Screening Requirements
                
                Banks are required to screen all transactions and customers against OFAC sanctions lists including:
                
                - Specially Designated Nationals (SDN) List
                - Sectoral Sanctions Identifications (SSI) List  
                - Foreign Sanctions Evaders (FSE) List
                - Non-SDN Palestinian Legislative Council (NS-PLC) List
                
                Screening must occur for:
                - All wire transfers regardless of amount
                - New account opening and customer onboarding
                - Periodic rescreening of existing customers
                - Trade finance transactions
                
                High-Risk Jurisdictions (requiring enhanced screening):
                - Iran, North Korea, Syria, Cuba
                - Russia (sectoral sanctions)
                - Certain regions: Crimea, Donetsk, Luhansk
                
                Compliance Requirements:
                - Real-time screening systems
                - Match investigation procedures
                - Blocking of prohibited transactions
                - Reporting to OFAC within 10 business days
                - Maintenance of blocked funds
                
                Penalties: Civil penalties up to $307,922 per violation or twice the amount of the underlying transaction.
                """,
                "metadata": {
                    "source": "OFAC Compliance Guidelines",
                    "type": "guidance",
                    "topic": "sanctions",
                    "effective_date": "2024-12-01", 
                    "authority": "OFAC"
                }
            },
            {
                "content": """
                WIRE TRANSFER COMPLIANCE - TRAVEL RULE
                
                Recordkeeping and Travel Rule for Funds Transfers (31 CFR 103.33)
                
                For funds transfers of $3,000 or more, banks must:
                
                Originating Bank Requirements:
                - Obtain and retain originator information:
                  * Name and address
                  * Account number or unique identifier
                  * Transaction amount and date
                
                Intermediary Bank Requirements:  
                - Retain all payment order information
                - Pass through originator and beneficiary information
                
                Beneficiary Bank Requirements:
                - Obtain and retain beneficiary information:
                  * Name and address
                  * Account number or unique identifier
                
                Enhanced Due Diligence for International Wires:
                - Wires to/from high-risk countries require additional verification
                - Large wire transfers (>$50,000) require management approval
                - Unusual wire transfer patterns must be investigated
                
                Suspicious Activity Reporting:
                Wire transfers involving:
                - Structured amounts to avoid reporting thresholds
                - Locations with no apparent business purpose
                - Round-dollar amounts with no commercial rationale
                - Rapid movement of funds through multiple accounts
                
                Must be reported via SAR within 30 days of detection.
                """,
                "metadata": {
                    "source": "31 CFR 103.33",
                    "type": "regulation",
                    "topic": "wire_transfers",
                    "effective_date": "2025-01-01",
                    "authority": "FinCEN"
                }
            },
            {
                "content": """
                CUSTOMER DUE DILIGENCE RULE
                
                31 CFR 1020.220 - Customer Due Diligence Requirements
                
                Effective May 11, 2018 (as amended 2024)
                
                Core Requirements:
                Banks must establish and maintain written procedures for:
                
                1. Customer Identification Program (CIP)
                   - Verify identity using documents, non-documentary methods, or combination
                   - Maintain records of information used to verify identity
                   - Compare customer against government lists (OFAC, etc.)
                
                2. Customer Due Diligence (CDD)
                   - Risk assessment based on customer type, products, services, geographic location
                   - Ongoing monitoring commensurate with risk
                   - Updating customer information periodically
                
                3. Beneficial Ownership for Legal Entity Customers
                   - Identify and verify beneficial owners (25% or more ownership)
                   - Identify single individual with significant control
                   - Maintain beneficial ownership certification
                
                Enhanced Due Diligence Required for:
                - Private banking accounts (>$1 million)
                - Correspondent banking relationships  
                - Politically exposed persons (PEPs)
                - High-risk geographic locations
                
                Risk Factors Requiring Enhanced Monitoring:
                - Unusual transaction patterns
                - Transactions inconsistent with business purpose
                - Large cash transactions
                - Multiple accounts with similar characteristics
                - Rapid movement of funds
                """,
                "metadata": {
                    "source": "31 CFR 1020.220",
                    "type": "regulation",
                    "topic": "customer_due_diligence",
                    "effective_date": "2024-05-15",
                    "authority": "FinCEN"
                }
            },
            {
                "content": """
                ABC BANK INTERNAL AML POLICY
                
                Policy Number: AML-001
                Last Updated: January 2025
                Next Review: January 2026
                
                TRANSACTION MONITORING THRESHOLDS
                
                Cash Transactions:
                - $10,000+: Automated CTR generation
                - $5,000-$9,999: Enhanced monitoring for structuring
                - Multiple cash deposits >$3,000 same day: Investigation required
                
                Wire Transfers:
                - International wires >$1,000: OFAC screening mandatory  
                - Domestic wires >$50,000: Enhanced review required
                - Same-day wires >$100,000: Senior approval required
                - High-risk country wires: Manager approval regardless of amount
                
                Customer Risk Rating Matrix:
                
                LOW RISK:
                - Domestic retail customers
                - Standard banking products only
                - Consistent transaction patterns
                - Complete KYC documentation
                
                MEDIUM RISK:  
                - Small business customers
                - International wire activity
                - Cash-intensive businesses
                - Customers from medium-risk countries
                
                HIGH RISK:
                - Politically exposed persons (PEPs)
                - Money service businesses
                - Customers from high-risk jurisdictions
                - Complex beneficial ownership structures
                - Unusual transaction patterns
                
                MONITORING REQUIREMENTS BY RISK LEVEL:
                - Low Risk: Quarterly review
                - Medium Risk: Monthly review  
                - High Risk: Weekly review
                - PEPs: Transaction-by-transaction review
                
                ESCALATION PROCEDURES:
                Suspicious Activity: Report to Compliance Officer within 24 hours
                Potential SAR: File within 30 days of detection
                OFAC Match: Immediate escalation and transaction blocking
                """,
                "metadata": {
                    "source": "ABC Bank Policy Manual",
                    "type": "internal_policy",
                    "topic": "transaction_monitoring",
                    "effective_date": "2025-01-15",
                    "authority": "ABC Bank Compliance"
                }
            }
        ]
        
        documents = []
        for doc_data in fake_documents:
            doc = Document(
                text=doc_data["content"],
                metadata=doc_data["metadata"]
            )
            documents.append(doc)
        
        return documents
    
    async def search_documents(self, query: str, top_k: int = 5) -> List[DocumentSearchResult]:
        """Search documents using vector similarity"""
        if not self.query_engine:
            return []
            
        try:
            # Use retriever directly for more control
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=top_k
            )
            
            nodes = await asyncio.to_thread(retriever.retrieve, query)
            
            results = []
            for node in nodes:
                result = DocumentSearchResult(
                    content=node.text,
                    source=node.metadata.get("source", "Unknown"),
                    score=node.score if hasattr(node, 'score') else 0.0,
                    metadata=node.metadata
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Document search error: {e}")
            return []

class AsyncReActBankingAgent:
    """Async ReAct agent with proper document retrieval"""
    
    def __init__(self, document_store: RAGDocumentStore):
        self.document_store = document_store
        self.db_path = "async_banking_demo.db"
        self.max_iterations = 8
        self.setup_database()
    
    def setup_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT,
                amount REAL,
                transaction_type TEXT,
                description TEXT,
                country TEXT,
                risk_score REAL,
                timestamp TEXT,
                beneficiary_name TEXT,
                purpose_code TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT,
                customer_type TEXT,
                risk_rating TEXT,
                kyc_status TEXT,
                registration_country TEXT,
                is_pep INTEGER,
                last_review_date TEXT,
                account_opening_date TEXT
            )
        """)
        
        # Sample data
        sample_transactions = [
            ("TXN001", "CUST001", 25000.00, "WIRE", "Business payment to supplier", "US", 0.3, "2025-01-15 10:30:00", "ABC Suppliers Inc", "TRADE"),
            ("TXN002", "CUST002", 95000.00, "WIRE", "Real estate purchase payment", "US", 0.7, "2025-01-15 11:45:00", "Property Holdings LLC", "REAL_ESTATE"),
            ("TXN003", "CUST003", 500000.00, "WIRE", "International trade finance", "UAE", 0.9, "2025-01-15 14:20:00", "Global Trading Corp", "TRADE"),
            ("TXN004", "CUST001", 15000.00, "CASH", "Cash deposit", "US", 0.4, "2025-01-15 16:15:00", "N/A", "CASH"),
            ("TXN005", "CUST004", 150000.00, "WIRE", "Investment transfer", "Cayman Islands", 0.8, "2025-01-15 17:30:00", "Offshore Fund Ltd", "INVESTMENT"),
        ]
        
        sample_customers = [
            ("CUST001", "ABC Manufacturing LLC", "Business", "Low", "Complete", "US", 0, "2024-12-01", "2020-01-15"),
            ("CUST002", "John Smith", "Individual", "Medium", "Complete", "US", 0, "2024-11-15", "2019-05-20"),
            ("CUST003", "Global Trading Corp", "Business", "High", "Pending", "UAE", 1, "2024-10-01", "2023-03-10"),
            ("CUST004", "Offshore Investment Fund", "Business", "High", "Complete", "Cayman Islands", 1, "2024-09-15", "2022-08-05"),
        ]
        
        cursor.executemany("INSERT OR REPLACE INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_transactions)
        cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_customers)
        
        conn.commit()
        conn.close()
    
    async def get_transaction_data(self, transaction_id: str) -> Optional[Dict]:
        """Retrieve transaction and customer data"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        query = """
            SELECT t.*, c.name, c.customer_type, c.risk_rating, c.kyc_status, c.is_pep
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
            WHERE t.transaction_id = ?
        """
        
        result = await asyncio.to_thread(conn.execute, query, (transaction_id,))
        row = result.fetchone()
        conn.close()
        
        if not row:
            return None
            
        return {
            "transaction_id": row[0],
            "customer_id": row[1],
            "amount": row[2],
            "transaction_type": row[3],
            "description": row[4],
            "country": row[5],
            "risk_score": row[6],
            "timestamp": row[7],
            "beneficiary_name": row[8],
            "purpose_code": row[9],
            "customer_name": row[10],
            "customer_type": row[11],
            "risk_rating": row[12],
            "kyc_status": row[13],
            "is_pep": bool(row[14])
        }
    
    async def _execute_action(self, action: AgentAction) -> AgentObservation:
        """Execute agent action asynchronously"""
        
        try:
            if action.action_type == ActionType.SEARCH_DOCUMENTS:
                query = action.parameters.get("query", "")
                docs = await self.document_store.search_documents(query, top_k=3)
                
                result = {
                    "documents_found": len(docs),
                    "relevant_content": [
                        {
                            "source": doc.source,
                            "content": doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                            "topic": doc.metadata.get("topic", "general"),
                            "authority": doc.metadata.get("authority", "Unknown")
                        } for doc in docs
                    ]
                }
                
                sources = [doc.source for doc in docs]
                return AgentObservation(result=result, success=True, sources=sources)
            
            elif action.action_type == ActionType.CHECK_CUSTOMER:
                customer_id = action.parameters.get("customer_id", "")
                customer_data = await self._get_customer_data(customer_id)
                return AgentObservation(result=customer_data, success=True)
            
            elif action.action_type == ActionType.CALCULATE_RISK:
                risk_result = await self._calculate_risk_async(action.parameters)
                return AgentObservation(result=risk_result, success=True)
            
            elif action.action_type == ActionType.CHECK_SANCTIONS:
                sanctions_result = await self._check_sanctions_async(action.parameters.get("entity_name", ""))
                return AgentObservation(result=sanctions_result, success=True)
            
            elif action.action_type == ActionType.VALIDATE_TRANSACTION:
                validation_result = await self._validate_transaction_async(action.parameters)
                return AgentObservation(result=validation_result, success=True)
            
            elif action.action_type == ActionType.FINAL_DECISION:
                decision_result = await self._make_final_decision_async(action.parameters)
                return AgentObservation(result=decision_result, success=True)
            
            else:
                return AgentObservation(
                    result=None,
                    success=False,
                    error_message=f"Unknown action type: {action.action_type}"
                )
                
        except Exception as e:
            return AgentObservation(
                result=None,
                success=False,
                error_message=str(e)
            )
    
    async def _get_customer_data(self, customer_id: str) -> Dict:
        """Get customer information"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        query = "SELECT * FROM customers WHERE customer_id = ?"
        result = await asyncio.to_thread(conn.execute, query, (customer_id,))
        row = result.fetchone()
        conn.close()
        
        if not row:
            return {"error": "Customer not found"}
            
        return {
            "customer_id": row[0],
            "name": row[1],
            "customer_type": row[2],
            "risk_rating": row[3],
            "kyc_status": row[4],
            "registration_country": row[5],
            "is_pep": bool(row[6]),
            "last_review_date": row[7],
            "account_opening_date": row[8]
        }
    
    async def _calculate_risk_async(self, parameters: Dict) -> Dict:
        """Calculate risk score asynchronously"""
        # Simulate async risk calculation
        await asyncio.sleep(0.1)
        
        risk_score = 0.0
        risk_factors = []
        
        amount = parameters.get("amount", 0)
        country = parameters.get("country", "")
        is_pep = parameters.get("is_pep", False)
        
        # Risk calculation logic
        if amount > 500000:
            risk_score += 0.4
            risk_factors.append("Very large transaction amount")
        elif amount > 100000:
            risk_score += 0.3
            risk_factors.append("Large transaction amount")
        
        high_risk_countries = ["UAE", "Nigeria", "Myanmar", "Iran", "North Korea"]
        if country in high_risk_countries:
            risk_score += 0.3
            risk_factors.append(f"High-risk jurisdiction: {country}")
        
        if is_pep:
            risk_score += 0.2
            risk_factors.append("Politically Exposed Person")
        
        return {
            "calculated_risk_score": min(risk_score, 1.0),
            "risk_factors": risk_factors,
            "risk_level": "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW"
        }
    
    async def _check_sanctions_async(self, entity_name: str) -> Dict:
        """Check sanctions lists asynchronously"""
        await asyncio.sleep(0.1)
        
        # Simulated sanctions screening
        sanctioned_entities = ["Global Trading Corp", "Sanctioned Entity Ltd", "Bad Actor Inc"]
        
        is_sanctioned = entity_name in sanctioned_entities
        
        return {
            "entity_name": entity_name,
            "is_sanctioned": is_sanctioned,
            "sanctions_lists_checked": ["OFAC SDN", "EU Consolidated", "UN 1267"],
            "match_details": [{"list": "OFAC SDN", "match_type": "exact"}] if is_sanctioned else []
        }
    
    async def _validate_transaction_async(self, parameters: Dict) -> Dict:
        """Validate transaction against business rules"""
        await asyncio.sleep(0.1)
        
        violations = []
        warnings = []
        
        amount = parameters.get("amount", 0)
        transaction_type = parameters.get("transaction_type", "")
        kyc_status = parameters.get("kyc_status", "")
        
        # Business rule validation
        if transaction_type == "CASH" and amount >= 10000:
            violations.append("Cash transaction exceeds $10,000 - CTR filing required")
        
        if transaction_type == "WIRE" and amount >= 50000:
            warnings.append("Wire transfer exceeds review threshold")
        
        if kyc_status != "Complete":
            violations.append("KYC documentation incomplete")
        
        return {
            "violations": violations,
            "warnings": warnings,
            "is_compliant": len(violations) == 0,
            "rules_checked": ["CTR Requirements", "Wire Transfer Limits", "KYC Compliance"]
        }
    
    async def _make_final_decision_async(self, parameters: Dict) -> Dict:
        """Make final compliance decision"""
        await asyncio.sleep(0.1)
        
        requires_review = False
        compliance_issues = []
        recommendations = []
        
        # Decision logic based on gathered information
        risk_level = parameters.get("risk_level", "LOW")
        violations = parameters.get("violations", [])
        is_sanctioned = parameters.get("is_sanctioned", False)
        
        if is_sanctioned:
            requires_review = True
            compliance_issues.append("SANCTIONS MATCH - Do not process transaction")
            recommendations.append("Block transaction immediately and report to OFAC")
        
        if violations:
            requires_review = True
            compliance_issues.extend(violations)
            recommendations.append("Address all compliance violations before processing")
        
        if risk_level == "HIGH":
            requires_review = True
            recommendations.append("Enhanced due diligence and senior approval required")
        
        return {
            "requires_review": requires_review,
            "compliance_issues": compliance_issues,
            "recommendations": recommendations,
            "final_decision": "BLOCK" if is_sanctioned else "REVIEW" if requires_review else "APPROVE"
        }
    
    async def _generate_reasoning_async(self, step_number: int, context: Dict, previous_observations: List[AgentObservation]) -> Tuple[str, AgentAction]:
        """Generate reasoning using OpenAI asynchronously"""
        
        context_str = json.dumps(context, indent=2, default=str)
        
        # Build observation summary
        obs_summary = []
        for i, obs in enumerate(previous_observations[-3:]):
            if obs.success:
                obs_summary.append(f"Step {len(previous_observations)-2+i}: Success - {obs.result}")
            else:
                obs_summary.append(f"Step {len(previous_observations)-2+i}: Failed - {obs.error_message}")
        
        reasoning_prompt = f"""
        You are a banking compliance ReAct agent. Analyze transactions systematically using available actions.
        
        Transaction Context:
        {context_str}
        
        Previous Steps:
        {chr(10).join(obs_summary) if obs_summary else "This is the first step"}
        
        Current Step: {step_number}
        
        Available Actions:
        1. search_documents(query) - Search regulatory documents and policies
        2. check_customer(customer_id) - Get customer profile and risk rating
        3. calculate_risk(parameters) - Calculate transaction risk score
        4. check_sanctions(entity_name) - Screen against sanctions lists
        5. validate_transaction(parameters) - Check compliance rules
        6. final_decision(parameters) - Make compliance determination
        
        Think about what information you need and choose the most appropriate action.
        
        Format:
        THOUGHT: [Your reasoning about what to do next]
        ACTION: [action_name]
        PARAMETERS: [JSON parameters]
        """
        
        try:
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a banking compliance expert. Think step by step."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                max_tokens=600,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            import re
            thought_match = re.search(r'THOUGHT:\s*(.*?)(?=ACTION:|$)', response_text, re.DOTALL)
            action_match = re.search(r'ACTION:\s*(\w+)', response_text)
            params_match = re.search(r'PARAMETERS:\s*(\{.*?\})', response_text, re.DOTALL)
            
            if not thought_match or not action_match:
                raise ValueError("Failed to parse LLM response")
            
            thought = thought_match.group(1).strip()
            action_name = action_match.group(1).strip()
            
            try:
                parameters = json.loads(params_match.group(1)) if params_match else {}
            except json.JSONDecodeError:
                parameters = {}
            
            # Map action to enum
            action_type_map = {
                "search_documents": ActionType.SEARCH_DOCUMENTS,
                "check_customer": ActionType.CHECK_CUSTOMER,
                "calculate_risk": ActionType.CALCULATE_RISK,
                "check_sanctions": ActionType.CHECK_SANCTIONS,
                "validate_transaction": ActionType.VALIDATE_TRANSACTION,
                "final_decision": ActionType.FINAL_DECISION
            }
            
            action_type = action_type_map.get(action_name)
            if not action_type:
                raise ValueError(f"Unknown action: {action_name}")
            
            action = AgentAction(
                action_type=action_type,
                parameters=parameters,
                reasoning=thought
            )
            
            return thought, action
            
        except Exception as e:
            # Fallback reasoning
            return await self._fallback_reasoning_async(step_number, context)
    
    async def _fallback_reasoning_async(self, step_number: int, context: Dict) -> Tuple[str, AgentAction]:
        """Fallback reasoning if LLM fails"""
        
        tx_data = context.get("transaction_data", {})
        
        if step_number == 1:
            thought = "I need to search for relevant compliance documents first."
            action = AgentAction(
                action_type=ActionType.SEARCH_DOCUMENTS,
                parameters={"query": f"{tx_data.get('transaction_type', 'wire')} transfer compliance requirements"},
                reasoning=thought
            )
        elif step_number == 2:
            thought = "Now I should check the customer profile and risk rating."
            action = AgentAction(
                action_type=ActionType.CHECK_CUSTOMER,
                parameters={"customer_id": tx_data.get("customer_id", "")},
                reasoning=thought
            )
        elif step_number == 3:
            thought = "I need to screen the beneficiary for sanctions."
            action = AgentAction(
                action_type=ActionType.CHECK_SANCTIONS,
                parameters={"entity_name": tx_data.get("beneficiary_name", "")},
                reasoning=thought
            )
        else:
            thought = "I have enough information to make a final decision."
            action = AgentAction(
                action_type=ActionType.FINAL_DECISION,
                parameters={"risk_level": "MEDIUM", "violations": [], "is_sanctioned": False},
                reasoning=thought
            )
        
        return thought, action
    
    async def analyze_transaction(self, transaction_id: str) -> Tuple[ComplianceResult, List[ReActStep]]:
        """Main ReAct analysis loop - async version"""
        
        start_time = datetime.now()
        
        # Get transaction data
        tx_data = await self.get_transaction_data(transaction_id)
        if not tx_data:
            raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found")
        
        # Initialize ReAct loop
        react_steps = []
        observations = []
        context = {"transaction_data": tx_data}
        analysis_id = str(uuid.uuid4())
        
        print(f"Starting async ReAct analysis for {transaction_id}")
        
        # Main reasoning loop
        for step in range(1, self.max_iterations + 1):
            
            # Generate reasoning and action
            thought, action = await self._generate_reasoning_async(step, context, observations)
            
            print(f"Step {step}: {action.action_type.value}")
            
            # Execute action
            observation = await self._execute_action(action)
            observations.append(observation)
            
            # Record step
            react_step = ReActStep(
                step_number=step,
                thought=thought,
                action=action,
                observation=observation,
                timestamp=datetime.now()
            )
            react_steps.append(react_step)
            
            # Update context
            if observation.success and observation.result:
                context[f"step_{step}_result"] = observation.result
            
            # Stop if final decision made
            if action.action_type == ActionType.FINAL_DECISION and observation.success:
                print(f"Analysis complete after {step} steps")
                break
        
        # Compile results
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Extract final information
        compliance_issues = []
        recommendations = []
        regulatory_references = []
        requires_review = False
        risk_level = "MEDIUM"
        
        # Process all observations
        for obs in observations:
            if obs.success and isinstance(obs.result, dict):
                if "compliance_issues" in obs.result:
                    compliance_issues.extend(obs.result["compliance_issues"])
                if "violations" in obs.result:
                    compliance_issues.extend(obs.result["violations"])
                if "recommendations" in obs.result:
                    recommendations.extend(obs.result["recommendations"])
                if "requires_review" in obs.result:
                    requires_review = obs.result["requires_review"]
                if "risk_level" in obs.result:
                    risk_level = obs.result["risk_level"]
                if obs.sources:
                    regulatory_references.extend(obs.sources)
        
        # Remove duplicates
        compliance_issues = list(set(compliance_issues))[:5]
        recommendations = list(set(recommendations))[:5]
        regulatory_references = list(set(regulatory_references))[:3]
        
        result = ComplianceResult(
            transaction_id=transaction_id,
            analysis_id=analysis_id,
            risk_level=risk_level,
            requires_review=requires_review,
            compliance_issues=compliance_issues,
            recommendations=recommendations,
            regulatory_references=regulatory_references,
            confidence_score=0.9,
            analysis_summary=f"Async ReAct analysis completed in {len(react_steps)} steps with document retrieval and systematic compliance review",
            total_steps=len(react_steps),
            processing_time_ms=int(processing_time)
        )
        
        return result, react_steps

# Global instances
document_store = RAGDocumentStore()
react_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    print("Initializing document store...")
    await document_store.initialize()
    
    global react_agent
    react_agent = AsyncReActBankingAgent(document_store)
    print("ReAct agent initialized")
    
    yield
    
    # Shutdown
    print("Shutting down...")

# FastAPI app
app = FastAPI(
    title="RE:Agent Banking Compliance API",
    description="ReAct AI Agent for Banking Compliance with Document Retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for analysis results (use Redis in production)
analysis_cache = {}
trace_cache = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "RE:Agent Banking Compliance API",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze/{transaction_id}")
async def analyze_transaction(
    transaction_id: str,
    background_tasks: BackgroundTasks
) -> ComplianceResult:
    """Analyze a transaction for compliance using ReAct agent"""
    
    if not react_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result, react_steps = await react_agent.analyze_transaction(transaction_id)
        
        # Cache results
        analysis_cache[result.analysis_id] = result
        trace_cache[result.analysis_id] = react_steps
        
        # Clean up old cache entries in background
        background_tasks.add_task(cleanup_old_cache_entries)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analysis/{analysis_id}/trace")
async def get_reasoning_trace(analysis_id: str) -> List[ReActTrace]:
    """Retrieve the detailed ReAct reasoning trace for an analysis"""
    
    if analysis_id not in trace_cache:
        raise HTTPException(status_code=404, detail="Analysis trace not found")
    
    react_steps = trace_cache[analysis_id]
    
    trace_results = []
    for step in react_steps:
        trace_result = ReActTrace(
            step_number=step.step_number,
            thought=step.thought,
            action_type=step.action.action_type.value,
            action_parameters=step.action.parameters,
            observation_success=step.observation.success,
            observation_result=step.observation.result,
            sources=step.observation.sources,
            timestamp=step.timestamp.isoformat()
        )
        trace_results.append(trace_result)
    
    return trace_results

@app.get("/search")
async def search_documents(
    query: str = Query(..., description="Search query for compliance documents"),
    top_k: int = Query(5, description="Number of results to return")
) -> List[DocumentSearchResult]:
    """Search compliance documents directly"""
    
    if not document_store:
        raise HTTPException(status_code=503, detail="Document store not initialized")
    
    try:
        results = await document_store.search_documents(query, top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/transactions")
async def list_available_transactions():
    """List available sample transactions for testing"""
    
    transactions = [
        {"id": "TXN001", "description": "$25K Business Payment", "risk": "Low"},
        {"id": "TXN002", "description": "$95K Real Estate Purchase", "risk": "Medium"},
        {"id": "TXN003", "description": "$500K International Trade (PEP)", "risk": "High"},
        {"id": "TXN004", "description": "$15K Cash Deposit", "risk": "Medium"},
        {"id": "TXN005", "description": "$150K Offshore Investment (PEP)", "risk": "High"}
    ]
    
    return {"transactions": transactions}

@app.post("/batch-analyze")
async def batch_analyze_transactions(transaction_ids: List[str]) -> List[ComplianceResult]:
    """Analyze multiple transactions in parallel"""
    
    if not react_agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    if len(transaction_ids) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 transactions per batch")
    
    try:
        # Process transactions in parallel
        tasks = [react_agent.analyze_transaction(tx_id) for tx_id in transaction_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                error_result = ComplianceResult(
                    transaction_id=transaction_ids[i],
                    analysis_id=str(uuid.uuid4()),
                    risk_level="UNKNOWN",
                    requires_review=True,
                    compliance_issues=[f"Analysis failed: {str(result)}"],
                    recommendations=["Manual review required due to system error"],
                    regulatory_references=[],
                    confidence_score=0.0,
                    analysis_summary="Analysis failed due to system error",
                    total_steps=0,
                    processing_time_ms=0
                )
                batch_results.append(error_result)
            else:
                compliance_result, react_steps = result
                batch_results.append(compliance_result)
                
                # Cache successful results
                analysis_cache[compliance_result.analysis_id] = compliance_result
                trace_cache[compliance_result.analysis_id] = react_steps
        
        return batch_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

async def cleanup_old_cache_entries():
    """Background task to clean up old cache entries"""
    current_time = datetime.now()
    
    # Remove entries older than 1 hour
    to_remove = []
    for analysis_id, result in analysis_cache.items():
        # This would need proper timestamp tracking in production
        to_remove.append(analysis_id)  # Simplified for demo
    
    # Keep only last 100 entries for demo
    if len(analysis_cache) > 100:
        oldest_ids = list(analysis_cache.keys())[:-100]
        for analysis_id in oldest_ids:
            analysis_cache.pop(analysis_id, None)
            trace_cache.pop(analysis_id, None)

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_react_rag_system:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )