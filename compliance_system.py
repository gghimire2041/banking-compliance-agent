# Simplified RE:Agent Prototype - Minimal Dependencies Version
# This version uses only basic libraries to avoid installation issues

import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
import openai

# Simple Streamlit alternative using basic web interface
try:
    import streamlit as st
    USE_STREAMLIT = True
except ImportError:
    USE_STREAMLIT = False
    print("Streamlit not available. Using command line interface.")

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
if OPENAI_API_KEY == "your-openai-api-key-here":
    OPENAI_API_KEY = input("Enter your OpenAI API key: ")

openai.api_key = OPENAI_API_KEY

@dataclass
class ComplianceResult:
    transaction_id: str
    risk_level: str
    requires_review: bool
    compliance_issues: List[str]
    recommendations: List[str]
    regulatory_references: List[str]
    confidence_score: float
    analysis_summary: str

class SimpleBankingComplianceSystem:
    def __init__(self):
        self.db_path = "simple_bank_demo.db"
        self.compliance_docs = self._load_compliance_knowledge()
        self.setup_database()
        
    def _load_compliance_knowledge(self):
        """Load compliance knowledge base (simulated document retrieval)"""
        return {
            "aml_policy": """
            BANK OF AI - ANTI-MONEY LAUNDERING POLICY
            
            Key Requirements:
            - Transactions >$10,000: Enhanced due diligence required
            - Wire transfers >$50,000: Additional verification needed
            - High-risk countries (UAE, Nigeria, Myanmar): Manager approval required
            - PEP customers: Senior approval for all transactions
            - Customers with 'High' risk rating: Quarterly review needed
            - Cash transactions >$10,000: CTR filing required within 24 hours
            - Suspicious transactions: Report within 24 hours
            """,
            
            "wire_transfer_rules": """
            WIRE TRANSFER COMPLIANCE MANUAL
            
            International Wire Transfers:
            - All international wires >$3,000: OFAC screening required
            - Complete beneficiary information mandatory
            - Trade finance: Requires commercial invoices, bills of lading
            
            Risk Assessment Guidelines:
            - Risk Score >0.8: Automatic hold for manual review
            - Risk Score 0.5-0.8: Enhanced monitoring required  
            - Risk Score <0.5: Standard processing acceptable
            """,
            
            "regulatory_updates": """
            BSA/AML REQUIREMENTS 2025
            
            New Requirements:
            - Enhanced CDD for legal entity customers
            - Beneficial ownership verification for business accounts
            - Real-time monitoring for high-risk customers
            
            Penalties: Civil penalties up to $75,000 per violation
            Best Practices: Risk-based customer due diligence approach
            """
        }
    
    def setup_database(self):
        """Initialize SQLite database with sample data"""
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
                is_pep INTEGER
            )
        """)
        
        # Sample transactions
        sample_transactions = [
            ("TXN001", "CUST001", 25000.00, "WIRE", "Business payment to supplier", "US", 0.3, "2025-01-15 10:30:00"),
            ("TXN002", "CUST002", 95000.00, "WIRE", "Real estate purchase payment", "US", 0.7, "2025-01-15 11:45:00"),
            ("TXN003", "CUST003", 500000.00, "WIRE", "International trade finance", "UAE", 0.9, "2025-01-15 14:20:00"),
            ("TXN004", "CUST001", 8500.00, "ATM", "Cash withdrawal", "US", 0.2, "2025-01-15 16:15:00"),
            ("TXN005", "CUST004", 150000.00, "WIRE", "Investment transfer", "Cayman Islands", 0.8, "2025-01-15 17:30:00"),
        ]
        
        sample_customers = [
            ("CUST001", "ABC Manufacturing LLC", "Business", "Low", "Complete", "US", 0),
            ("CUST002", "John Smith", "Individual", "Medium", "Complete", "US", 0),
            ("CUST003", "Global Trading Corp", "Business", "High", "Pending", "UAE", 1),
            ("CUST004", "Offshore Investment Fund", "Business", "High", "Complete", "Cayman Islands", 1),
        ]
        
        cursor.executemany("INSERT OR REPLACE INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", sample_transactions)
        cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?)", sample_customers)
        
        conn.commit()
        conn.close()
        print("✅ Database initialized with sample data")
    
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
    
    def _retrieve_relevant_policies(self, tx_data: Dict) -> str:
        """Simple document retrieval simulation (replaces complex RAG)"""
        relevant_content = []
        
        # Always include AML policy
        relevant_content.append(self.compliance_docs["aml_policy"])
        
        # Include wire transfer rules for wire transfers
        if tx_data["transaction_type"] == "WIRE":
            relevant_content.append(self.compliance_docs["wire_transfer_rules"])
        
        # Include regulatory updates for high-risk scenarios
        if tx_data["amount"] > 50000 or tx_data["is_pep"] or tx_data["risk_score"] > 0.7:
            relevant_content.append(self.compliance_docs["regulatory_updates"])
        
        return "\n\n".join(relevant_content)
    
    def analyze_transaction(self, transaction_id: str) -> ComplianceResult:
        """Main compliance analysis function using OpenAI"""
        
        # Get transaction data
        tx_data = self.get_transaction_data(transaction_id)
        if not tx_data:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        # Retrieve relevant policies
        relevant_policies = self._retrieve_relevant_policies(tx_data)
        
        # Create analysis prompt
        analysis_prompt = f"""
        You are a banking compliance expert analyzing a transaction. Based on the transaction details and compliance policies provided, analyze the transaction for compliance issues.

        TRANSACTION DETAILS:
        - ID: {tx_data['transaction_id']}
        - Amount: ${tx_data['amount']:,.2f}
        - Type: {tx_data['transaction_type']}
        - Description: {tx_data['description']}
        - Country: {tx_data['country']}
        - Risk Score: {tx_data['risk_score']}
        
        CUSTOMER INFORMATION:
        - Name: {tx_data['customer_name']}
        - Type: {tx_data['customer_type']}
        - Risk Rating: {tx_data['risk_rating']}
        - KYC Status: {tx_data['kyc_status']}
        - PEP Status: {'Yes' if tx_data['is_pep'] else 'No'}
        
        RELEVANT COMPLIANCE POLICIES:
        {relevant_policies}
        
        Please provide a detailed analysis in the following JSON format:
        {{
            "compliance_issues": ["list of specific compliance issues found"],
            "recommendations": ["list of recommended actions"],
            "regulatory_references": ["relevant policy sections"],
            "analysis_summary": "brief summary of overall compliance status"
        }}
        
        Be specific about policy violations and required actions.
        """
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a banking compliance expert. Provide thorough, accurate compliance analysis."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )
            
            # Parse response
            ai_analysis = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                parsed_analysis = json.loads(ai_analysis)
            except json.JSONDecodeError:
                # Fallback parsing if JSON format is not perfect
                parsed_analysis = self._parse_text_response(ai_analysis)
            
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            # Fallback to rule-based analysis
            parsed_analysis = self._fallback_analysis(tx_data)
        
        # Determine risk level and review requirement
        requires_review = self._determine_review_requirement(tx_data)
        risk_level = self._calculate_risk_level(tx_data)
        
        return ComplianceResult(
            transaction_id=transaction_id,
            risk_level=risk_level,
            requires_review=requires_review,
            compliance_issues=parsed_analysis.get("compliance_issues", []),
            recommendations=parsed_analysis.get("recommendations", []),
            regulatory_references=parsed_analysis.get("regulatory_references", []),
            confidence_score=0.85,
            analysis_summary=parsed_analysis.get("analysis_summary", "Analysis completed")
        )
    
    def _parse_text_response(self, response_text: str) -> Dict:
        """Parse non-JSON response from AI"""
        lines = response_text.split('\n')
        
        issues = []
        recommendations = []
        references = []
        
        for line in lines:
            line = line.strip()
            if any(word in line.lower() for word in ['violation', 'issue', 'problem', 'concern']):
                issues.append(line)
            elif any(word in line.lower() for word in ['recommend', 'should', 'must', 'required']):
                recommendations.append(line)
            elif any(word in line.lower() for word in ['policy', 'regulation', 'requirement', 'rule']):
                references.append(line)
        
        return {
            "compliance_issues": issues[:5],  # Limit to top 5
            "recommendations": recommendations[:5],
            "regulatory_references": references[:3],
            "analysis_summary": response_text[:200] + "..." if len(response_text) > 200 else response_text
        }
    
    def _fallback_analysis(self, tx_data: Dict) -> Dict:
        """Rule-based fallback analysis if OpenAI fails"""
        issues = []
        recommendations = []
        references = []
        
        # Check common compliance rules
        if tx_data["amount"] > 10000:
            issues.append(f"Large transaction ${tx_data['amount']:,.2f} requires enhanced due diligence")
            recommendations.append("Perform enhanced customer due diligence verification")
            references.append("AML Policy Section 1: Transaction Monitoring")
        
        if tx_data["is_pep"]:
            issues.append("Transaction involves Politically Exposed Person (PEP)")
            recommendations.append("Obtain senior management approval before processing")
            references.append("AML Policy Section 3: Customer Due Diligence")
        
        if tx_data["risk_score"] > 0.7:
            issues.append(f"High risk score ({tx_data['risk_score']}) detected")
            recommendations.append("Manual review required before processing")
            references.append("Wire Transfer Manual Chapter 4: Risk Assessment")
        
        if tx_data["country"] in ["UAE", "Nigeria", "Myanmar", "Cayman Islands"]:
            issues.append(f"Transaction involves high-risk jurisdiction: {tx_data['country']}")
            recommendations.append("Additional enhanced due diligence required")
            references.append("AML Policy Section 2: High-Risk Countries")
        
        return {
            "compliance_issues": issues,
            "recommendations": recommendations,
            "regulatory_references": references,
            "analysis_summary": f"Rule-based analysis completed for transaction {tx_data['transaction_id']}"
        }
    
    def _determine_review_requirement(self, tx_data: Dict) -> bool:
        """Business logic to determine if transaction requires review"""
        
        if tx_data['amount'] > 50000:  # Large transactions
            return True
        if tx_data['risk_score'] > 0.7:  # High risk score
            return True
        if tx_data['is_pep']:  # PEP customer
            return True
        if tx_data['country'] in ['UAE', 'Nigeria', 'Myanmar', 'Cayman Islands']:  # High-risk countries
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

def command_line_interface():
    """Simple command line interface if Streamlit is not available"""
    
    system = SimpleBankingComplianceSystem()
    
    print("\n" + "="*60)
    print("🏦 RE:Agent Prototype - Banking Compliance System")
    print("="*60)
    
    while True:
        print("\nAvailable Transactions:")
        print("1. TXN001 - $25K Business Payment (Low Risk)")
        print("2. TXN002 - $95K Real Estate Purchase (Medium Risk)")
        print("3. TXN003 - $500K International Trade - UAE PEP (High Risk)")
        print("4. TXN004 - $8.5K ATM Withdrawal (Low Risk)")
        print("5. TXN005 - $150K Offshore Investment - Cayman Islands PEP (High Risk)")
        print("0. Exit")
        
        choice = input("\nSelect transaction to analyze (1-5) or 0 to exit: ")
        
        if choice == "0":
            break
            
        transaction_map = {
            "1": "TXN001",
            "2": "TXN002", 
            "3": "TXN003",
            "4": "TXN004",
            "5": "TXN005"
        }
        
        if choice in transaction_map:
            transaction_id = transaction_map[choice]
            
            print(f"\n🔍 Analyzing {transaction_id}...")
            print("-" * 50)
            
            try:
                result = system.analyze_transaction(transaction_id)
                tx_data = system.get_transaction_data(transaction_id)
                
                # Display results
                print(f"Transaction: {tx_data['customer_name']} - ${tx_data['amount']:,.2f}")
                print(f"Risk Level: {result.risk_level}")
                print(f"Review Required: {'YES' if result.requires_review else 'NO'}")
                print(f"Confidence: {result.confidence_score:.0%}")
                
                if result.compliance_issues:
                    print(f"\n⚠️ Compliance Issues ({len(result.compliance_issues)}):")
                    for i, issue in enumerate(result.compliance_issues, 1):
                        print(f"  {i}. {issue}")
                
                if result.recommendations:
                    print(f"\n📋 Recommendations ({len(result.recommendations)}):")
                    for i, rec in enumerate(result.recommendations, 1):
                        print(f"  {i}. {rec}")
                
                if result.regulatory_references:
                    print(f"\n📚 Regulatory References:")
                    for ref in result.regulatory_references:
                        print(f"  • {ref}")
                
                print(f"\n💡 Summary: {result.analysis_summary}")
                
            except Exception as e:
                print(f"❌ Error analyzing transaction: {e}")
        else:
            print("Invalid choice. Please select 1-5 or 0 to exit.")

def streamlit_interface():
    """Streamlit web interface"""
    
    st.set_page_config(page_title="RE:Agent Prototype", page_icon="🏦", layout="wide")
    
    st.title("🏦 RE:Agent Prototype - Banking Compliance System")
    st.markdown("*AI-Powered Transaction Compliance Review System*")
    
    # Initialize system
    if 'compliance_system' not in st.session_state:
        with st.spinner("Initializing compliance system..."):
            st.session_state.compliance_system = SimpleBankingComplianceSystem()
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Transaction Analysis")
        
        transaction_options = {
            "TXN001": "TXN001 - $25K Business Payment",
            "TXN002": "TXN002 - $95K Real Estate Purchase", 
            "TXN003": "TXN003 - $500K International Trade (PEP)",
            "TXN004": "TXN004 - $8.5K ATM Withdrawal",
            "TXN005": "TXN005 - $150K Offshore Investment (PEP)"
        }
        
        transaction_id = st.selectbox(
            "Select Transaction:",
            list(transaction_options.keys()),
            format_func=lambda x: transaction_options[x]
        )
        
        if st.button("🔍 Analyze Transaction", type="primary"):
            with st.spinner("Analyzing transaction for compliance..."):
                try:
                    result = st.session_state.compliance_system.analyze_transaction(transaction_id)
                    st.session_state.analysis_result = result
                    st.session_state.selected_transaction = transaction_id
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.header("Compliance Analysis Results")
        
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            tx_data = st.session_state.compliance_system.get_transaction_data(st.session_state.selected_transaction)
            
            # Risk level indicator
            risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                st.metric("Risk Level", f"{risk_colors.get(result.risk_level, '⚪')} {result.risk_level}")
            with col2b:
                st.metric("Review Required", "YES" if result.requires_review else "NO")
            with col2c:
                st.metric("Confidence", f"{result.confidence_score:.0%}")
            
            # Transaction details
            st.subheader("Transaction Details")
            st.write(f"**Customer:** {tx_data['customer_name']} ({tx_data['customer_type']})")
            st.write(f"**Amount:** ${tx_data['amount']:,.2f} | **Type:** {tx_data['transaction_type']} | **Country:** {tx_data['country']}")
            st.write(f"**Risk Score:** {tx_data['risk_score']} | **PEP:** {'Yes' if tx_data['is_pep'] else 'No'} | **KYC:** {tx_data['kyc_status']}")
            
            # Compliance issues
            if result.compliance_issues:
                st.subheader("⚠️ Compliance Issues")
                for issue in result.compliance_issues:
                    st.warning(issue)
            
            # Recommendations
            if result.recommendations:
                st.subheader("📋 Recommendations")
                for rec in result.recommendations:
                    st.info(rec)
            
            # Summary
            st.subheader("Analysis Summary")
            st.write(result.analysis_summary)
            
        else:
            st.info("Select a transaction and click 'Analyze Transaction' to see results.")

def main():
    if USE_STREAMLIT and len(os.sys.argv) == 1:
        streamlit_interface()
    else:
        command_line_interface()

if __name__ == "__main__":
    main()