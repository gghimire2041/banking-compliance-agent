# RE:Agent Prototype - AI Banking Compliance System
# This prototype demonstrates intelligent transaction review using RAG and LLM

import os
import sqlite3
import pandas as pd
import streamlit as st
from datetime import datetime, date
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import openai

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
openai.api_key = OPENAI_API_KEY

# Configure LlamaIndex
Settings.llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    amount: float
    transaction_type: str
    description: str
    country: str
    risk_score: float
    timestamp: datetime

@dataclass
class ComplianceResult:
    transaction_id: str
    risk_level: str
    requires_review: bool
    compliance_issues: List[str]
    recommendations: List[str]
    regulatory_references: List[str]
    confidence_score: float

class BankingComplianceSystem:
    def __init__(self):
        self.db_path = "bank_ai_demo.db"
        self.vector_index = None
        self.setup_database()
        self.setup_documents()
        
    def setup_database(self):
        """Initialize SQLite database with sample banking data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT,
                amount REAL,
                transaction_type TEXT,
                description TEXT,
                country TEXT,
                risk_score REAL,
                timestamp TEXT
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
                is_pep BOOLEAN
            )
        """)
        
        # Sample data
        sample_transactions = [
            ("TXN001", "CUST001", 25000.00, "WIRE", "Business payment to supplier", "US", 0.3, "2025-01-15 10:30:00"),
            ("TXN002", "CUST002", 95000.00, "WIRE", "Real estate purchase", "US", 0.7, "2025-01-15 11:45:00"),
            ("TXN003", "CUST003", 500000.00, "WIRE", "International trade finance", "UAE", 0.9, "2025-01-15 14:20:00"),
            ("TXN004", "CUST001", 8500.00, "ATM", "Cash withdrawal", "US", 0.2, "2025-01-15 16:15:00"),
        ]
        
        sample_customers = [
            ("CUST001", "ABC Manufacturing LLC", "Business", "Low", "Complete", "US", False),
            ("CUST002", "John Smith", "Individual", "Medium", "Complete", "US", False),
            ("CUST003", "Global Trading Corp", "Business", "High", "Pending", "UAE", True),
        ]
        
        cursor.executemany("INSERT OR REPLACE INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sample_transactions)
        cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)", sample_customers)
        
        conn.commit()
        conn.close()
    
    def setup_documents(self):
        """Setup RAG system with compliance documents"""
        
        # Sample compliance documents (in practice, these would be loaded from files)
        documents = [
            Document(text="""
                BANK OF AI - ANTI-MONEY LAUNDERING POLICY
                
                Section 1: Transaction Monitoring
                All transactions exceeding $10,000 must undergo enhanced due diligence.
                Wire transfers above $50,000 require additional verification.
                
                Section 2: High-Risk Countries
                Transactions involving UAE, Nigeria, or Myanmar require manager approval.
                Enhanced monitoring required for countries on FATF grey list.
                
                Section 3: Customer Due Diligence
                PEP (Politically Exposed Person) transactions require senior approval.
                Customers with "High" risk rating need quarterly review.
                
                Section 4: Reporting Requirements
                Suspicious transactions must be reported within 24 hours.
                CTR filing required for cash transactions over $10,000.
                """, metadata={"source": "AML_Policy_2025.pdf", "type": "policy"}),
            
            Document(text="""
                BANK OF AI - WIRE TRANSFER COMPLIANCE MANUAL
                
                Chapter 3: International Wire Transfers
                - All international wires >$3,000 require OFAC screening
                - Beneficiary information must be complete and accurate
                - Correspondent banking relationships must be current
                
                Chapter 4: Risk Assessment
                Risk Score > 0.8: Automatic hold for manual review
                Risk Score 0.5-0.8: Enhanced monitoring required
                Risk Score < 0.5: Standard processing
                
                Chapter 5: Documentation
                Trade finance transactions require supporting documentation:
                - Commercial invoices
                - Bills of lading
                - Letter of credit
                """, metadata={"source": "Wire_Transfer_Manual.docx", "type": "manual"}),
            
            Document(text="""
                REGULATORY UPDATE - BSA/AML REQUIREMENTS 2025
                
                New Requirements Effective January 1, 2025:
                1. Enhanced CDD for legal entity customers
                2. Beneficial ownership verification for all new business accounts
                3. Real-time transaction monitoring for high-risk customers
                
                Penalties for Non-Compliance:
                - Civil penalties up to $75,000 per violation
                - Criminal prosecution for willful violations
                - Regulatory enforcement actions
                
                Best Practices:
                - Implement risk-based approach to customer due diligence
                - Maintain current and accurate customer information
                - Provide regular BSA/AML training to staff
                """, metadata={"source": "BSA_Update_2025.pdf", "type": "regulation"})
        ]
        
        # Create vector index
        node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        self.vector_index = VectorStoreIndex.from_documents(
            documents,
            node_parser=node_parser
        )
    
    def get_transaction_data(self, transaction_id: str) -> Optional[Dict]:
        """Retrieve transaction and customer data"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT t.*, c.name, c.customer_type, c.risk_rating, c.kyc_status, c.is_pep
            FROM transactions t
            JOIN customers c ON t.customer_id = c.customer_id
            WHERE t.transaction_id = ?
        """
        
        result = conn.execute(query, (transaction_id,)).fetchone()
        conn.close()
        
        if not result:
            return None
            
        return {
            "transaction_id": result[0],
            "customer_id": result[1],
            "amount": result[2],
            "transaction_type": result[3],
            "description": result[4],
            "country": result[5],
            "risk_score": result[6],
            "timestamp": result[7],
            "customer_name": result[8],
            "customer_type": result[9],
            "risk_rating": result[10],
            "kyc_status": result[11],
            "is_pep": bool(result[12])
        }
    
    def analyze_transaction(self, transaction_id: str) -> ComplianceResult:
        """Main compliance analysis function"""
        
        # Get transaction data
        tx_data = self.get_transaction_data(transaction_id)
        if not tx_data:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        # Create query engine
        query_engine = self.vector_index.as_query_engine(
            similarity_top_k=3,
            response_mode="tree_summarize"
        )
        
        # Build context for LLM
        context = f"""
        Transaction Details:
        - ID: {tx_data['transaction_id']}
        - Amount: ${tx_data['amount']:,.2f}
        - Type: {tx_data['transaction_type']}
        - Description: {tx_data['description']}
        - Country: {tx_data['country']}
        - Risk Score: {tx_data['risk_score']}
        
        Customer Information:
        - Name: {tx_data['customer_name']}
        - Type: {tx_data['customer_type']}
        - Risk Rating: {tx_data['risk_rating']}
        - KYC Status: {tx_data['kyc_status']}
        - PEP Status: {'Yes' if tx_data['is_pep'] else 'No'}
        """
        
        # Query compliance documents
        compliance_query = f"""
        Based on the following transaction details, what compliance requirements apply?
        {context}
        
        Please identify:
        1. Any policy violations or compliance issues
        2. Required approvals or additional steps
        3. Regulatory requirements that apply
        4. Risk mitigation recommendations
        """
        
        response = query_engine.query(compliance_query)
        
        # Determine risk level and review requirement
        requires_review = self._determine_review_requirement(tx_data)
        risk_level = self._calculate_risk_level(tx_data)
        
        # Parse compliance issues and recommendations
        issues, recommendations, references = self._parse_compliance_response(str(response))
        
        return ComplianceResult(
            transaction_id=transaction_id,
            risk_level=risk_level,
            requires_review=requires_review,
            compliance_issues=issues,
            recommendations=recommendations,
            regulatory_references=references,
            confidence_score=0.85  # In practice, this would be calculated
        )
    
    def _determine_review_requirement(self, tx_data: Dict) -> bool:
        """Business logic to determine if transaction requires review"""
        
        # High-risk scenarios requiring review
        if tx_data['amount'] > 50000:  # Large transactions
            return True
        if tx_data['risk_score'] > 0.7:  # High risk score
            return True
        if tx_data['is_pep']:  # PEP customer
            return True
        if tx_data['country'] in ['UAE', 'Nigeria', 'Myanmar']:  # High-risk countries
            return True
        if tx_data['kyc_status'] != 'Complete':  # Incomplete KYC
            return True
            
        return False
    
    def _calculate_risk_level(self, tx_data: Dict) -> str:
        """Calculate overall risk level"""
        
        risk_factors = 0
        
        if tx_data['amount'] > 100000:
            risk_factors += 2
        elif tx_data['amount'] > 50000:
            risk_factors += 1
            
        if tx_data['risk_score'] > 0.8:
            risk_factors += 2
        elif tx_data['risk_score'] > 0.5:
            risk_factors += 1
            
        if tx_data['is_pep']:
            risk_factors += 2
            
        if tx_data['risk_rating'] == 'High':
            risk_factors += 2
        elif tx_data['risk_rating'] == 'Medium':
            risk_factors += 1
            
        if risk_factors >= 4:
            return "HIGH"
        elif risk_factors >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _parse_compliance_response(self, response: str) -> tuple:
        """Parse LLM response into structured components"""
        
        # Simple parsing - in production, this would be more sophisticated
        issues = []
        recommendations = []
        references = []
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if 'violation' in line.lower() or 'issue' in line.lower():
                issues.append(line)
            elif 'recommend' in line.lower() or 'should' in line.lower():
                recommendations.append(line)
            elif 'policy' in line.lower() or 'regulation' in line.lower():
                references.append(line)
        
        return issues, recommendations, references

# Streamlit UI
def main():
    st.set_page_config(page_title="RE:Agent Prototype", page_icon="🏦", layout="wide")
    
    st.title("🏦 RE:Agent Prototype - Banking Compliance System")
    st.markdown("*AI-Powered Transaction Compliance Review System*")
    
    # Initialize system
    if 'compliance_system' not in st.session_state:
        with st.spinner("Initializing compliance system..."):
            st.session_state.compliance_system = BankingComplianceSystem()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("System Configuration")
        st.write("**Status:** ✅ Ready")
        st.write("**Documents Loaded:** 3 compliance documents")
        st.write("**Database:** Connected")
        
        st.header("Available Transactions")
        st.write("- TXN001: $25K Business Payment")
        st.write("- TXN002: $95K Real Estate")  
        st.write("- TXN003: $500K International Trade")
        st.write("- TXN004: $8.5K ATM Withdrawal")
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Transaction Input")
        
        transaction_id = st.selectbox(
            "Select Transaction ID:",
            ["TXN001", "TXN002", "TXN003", "TXN004"]
        )
        
        if st.button("🔍 Analyze Transaction", type="primary"):
            with st.spinner("Analyzing transaction for compliance..."):
                try:
                    result = st.session_state.compliance_system.analyze_transaction(transaction_id)
                    st.session_state.analysis_result = result
                except Exception as e:
                    st.error(f"Error analyzing transaction: {str(e)}")
    
    with col2:
        st.header("Compliance Analysis Results")
        
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            
            # Risk level indicator
            risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            st.metric(
                "Risk Level", 
                f"{risk_color.get(result.risk_level, '⚪')} {result.risk_level}",
                delta="Requires Review" if result.requires_review else "Auto-Approve"
            )
            
            # Compliance issues
            if result.compliance_issues:
                st.subheader("⚠️ Compliance Issues")
                for issue in result.compliance_issues:
                    st.warning(issue)
            else:
                st.success("✅ No compliance issues identified")
            
            # Recommendations
            if result.recommendations:
                st.subheader("📋 Recommendations")
                for rec in result.recommendations:
                    st.info(rec)
            
            # Regulatory references
            if result.regulatory_references:
                st.subheader("📚 Regulatory References")
                for ref in result.regulatory_references:
                    st.write(f"• {ref}")
            
            # Confidence score
            st.progress(result.confidence_score, text=f"Confidence: {result.confidence_score:.0%}")
            
        else:
            st.info("Select a transaction and click 'Analyze Transaction' to see results.")
    
    # Transaction details
    if 'analysis_result' in st.session_state:
        st.header("📊 Transaction Details")
        
        tx_data = st.session_state.compliance_system.get_transaction_data(
            st.session_state.analysis_result.transaction_id
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Amount", f"${tx_data['amount']:,.2f}")
        with col2:
            st.metric("Type", tx_data['transaction_type'])
        with col3:
            st.metric("Country", tx_data['country'])
        with col4:
            st.metric("Risk Score", f"{tx_data['risk_score']:.1f}")
        
        # Customer information
        st.subheader("Customer Information")
        customer_col1, customer_col2, customer_col3 = st.columns(3)
        with customer_col1:
            st.write(f"**Name:** {tx_data['customer_name']}")
            st.write(f"**Type:** {tx_data['customer_type']}")
        with customer_col2:
            st.write(f"**Risk Rating:** {tx_data['risk_rating']}")
            st.write(f"**KYC Status:** {tx_data['kyc_status']}")
        with customer_col3:
            st.write(f"**PEP Status:** {'Yes' if tx_data['is_pep'] else 'No'}")
            st.write(f"**Description:** {tx_data['description']}")

if __name__ == "__main__":
    main()
