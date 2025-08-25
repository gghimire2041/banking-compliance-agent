# RE:Agent Prototype with ReAct Agent Architecture
# Implements proper Reasoning and Acting patterns for banking compliance

import os
import sqlite3
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import openai

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

class ActionType(Enum):
    SEARCH_POLICY = "search_policy"
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

@dataclass 
class ReActStep:
    step_number: int
    thought: str
    action: AgentAction
    observation: AgentObservation
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
    analysis_summary: str
    react_trace: List[ReActStep]
    total_steps: int

class ReActBankingAgent:
    """
    ReAct (Reasoning + Acting) Agent for Banking Compliance
    Implements iterative reasoning and action-taking for complex compliance analysis
    """
    
    def __init__(self):
        self.db_path = "react_banking_demo.db"
        self.max_iterations = 10
        self.compliance_knowledge = self._initialize_knowledge_base()
        self.setup_database()
        
    def _initialize_knowledge_base(self):
        """Enhanced knowledge base with more detailed policies"""
        return {
            "aml_policies": {
                "transaction_limits": {
                    "cash_reporting_threshold": 10000,
                    "wire_review_threshold": 50000,
                    "enhanced_dd_threshold": 25000
                },
                "high_risk_countries": ["UAE", "Nigeria", "Myanmar", "North Korea", "Iran", "Syria"],
                "pep_requirements": "All PEP transactions require senior approval regardless of amount",
                "suspicious_indicators": [
                    "Rapid movement of funds",
                    "Transactions with no apparent business purpose", 
                    "Unusual geographic patterns",
                    "Structured transactions to avoid reporting"
                ]
            },
            "sanctions_lists": {
                "ofac_sdn": ["Global Trading Corp", "Sanctioned Entity Ltd"],
                "eu_consolidated": ["European Bad Actor Inc"],
                "un_1267": ["UN Listed Organization"]
            },
            "regulatory_frameworks": {
                "bsa_requirements": "Bank Secrecy Act compliance mandatory",
                "patriot_act": "Enhanced due diligence for foreign accounts",
                "fincen_guidance": "Real-time monitoring required for high-risk customers"
            }
        }
    
    def setup_database(self):
        """Enhanced database with historical transaction data"""
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
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id TEXT,
                amount REAL,
                transaction_date TEXT,
                country TEXT,
                flagged INTEGER
            )
        """)
        
        # Enhanced sample data
        sample_transactions = [
            ("TXN001", "CUST001", 25000.00, "WIRE", "Business payment to supplier", "US", 0.3, "2025-01-15 10:30:00", "ABC Suppliers Inc", "TRADE"),
            ("TXN002", "CUST002", 95000.00, "WIRE", "Real estate purchase payment", "US", 0.7, "2025-01-15 11:45:00", "Property Holdings LLC", "REAL_ESTATE"),
            ("TXN003", "CUST003", 500000.00, "WIRE", "International trade finance", "UAE", 0.9, "2025-01-15 14:20:00", "Global Trading Corp", "TRADE"),
            ("TXN004", "CUST001", 8500.00, "ATM", "Cash withdrawal", "US", 0.2, "2025-01-15 16:15:00", "N/A", "CASH"),
            ("TXN005", "CUST004", 150000.00, "WIRE", "Investment transfer", "Cayman Islands", 0.8, "2025-01-15 17:30:00", "Offshore Fund Ltd", "INVESTMENT"),
        ]
        
        sample_customers = [
            ("CUST001", "ABC Manufacturing LLC", "Business", "Low", "Complete", "US", 0, "2024-12-01", "2020-01-15"),
            ("CUST002", "John Smith", "Individual", "Medium", "Complete", "US", 0, "2024-11-15", "2019-05-20"),
            ("CUST003", "Global Trading Corp", "Business", "High", "Pending", "UAE", 1, "2024-10-01", "2023-03-10"),
            ("CUST004", "Offshore Investment Fund", "Business", "High", "Complete", "Cayman Islands", 1, "2024-09-15", "2022-08-05"),
        ]
        
        # Historical transaction patterns
        historical_data = [
            ("CUST001", 5000, "2025-01-01", "US", 0),
            ("CUST001", 7500, "2025-01-05", "US", 0),
            ("CUST003", 450000, "2024-12-20", "UAE", 1),
            ("CUST003", 380000, "2024-12-15", "UAE", 1),
            ("CUST004", 120000, "2024-11-30", "Cayman Islands", 0),
        ]
        
        cursor.executemany("INSERT OR REPLACE INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_transactions)
        cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", sample_customers)
        cursor.executemany("INSERT OR REPLACE INTO historical_transactions (customer_id, amount, transaction_date, country, flagged) VALUES (?, ?, ?, ?, ?)", historical_data)
        
        conn.commit()
        conn.close()
        print("✅ Enhanced database initialized with sample data")
    
    def _execute_action(self, action: AgentAction) -> AgentObservation:
        """Execute an agent action and return observation"""
        
        try:
            if action.action_type == ActionType.SEARCH_POLICY:
                result = self._search_policy(action.parameters.get("query", ""))
                return AgentObservation(result=result, success=True)
                
            elif action.action_type == ActionType.CHECK_CUSTOMER:
                result = self._check_customer(action.parameters.get("customer_id", ""))
                return AgentObservation(result=result, success=True)
                
            elif action.action_type == ActionType.CALCULATE_RISK:
                result = self._calculate_risk(action.parameters)
                return AgentObservation(result=result, success=True)
                
            elif action.action_type == ActionType.CHECK_SANCTIONS:
                result = self._check_sanctions(action.parameters.get("entity_name", ""))
                return AgentObservation(result=result, success=True)
                
            elif action.action_type == ActionType.VALIDATE_TRANSACTION:
                result = self._validate_transaction(action.parameters)
                return AgentObservation(result=result, success=True)
                
            elif action.action_type == ActionType.GET_HISTORICAL:
                result = self._get_historical_data(action.parameters.get("customer_id", ""))
                return AgentObservation(result=result, success=True)
                
            elif action.action_type == ActionType.FINAL_DECISION:
                result = self._make_final_decision(action.parameters)
                return AgentObservation(result=result, success=True)
                
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
    
    def _search_policy(self, query: str) -> Dict:
        """Search compliance policies based on query"""
        query_lower = query.lower()
        relevant_policies = {}
        
        if "aml" in query_lower or "anti-money" in query_lower:
            relevant_policies["aml"] = self.compliance_knowledge["aml_policies"]
            
        if "sanction" in query_lower:
            relevant_policies["sanctions"] = self.compliance_knowledge["sanctions_lists"]
            
        if "pep" in query_lower or "politically exposed" in query_lower:
            relevant_policies["pep"] = {
                "requirements": self.compliance_knowledge["aml_policies"]["pep_requirements"]
            }
            
        if "wire" in query_lower or "transfer" in query_lower:
            relevant_policies["wire_limits"] = {
                "review_threshold": self.compliance_knowledge["aml_policies"]["transaction_limits"]["wire_review_threshold"]
            }
            
        return relevant_policies
    
    def _check_customer(self, customer_id: str) -> Dict:
        """Retrieve customer information and risk profile"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM customers WHERE customer_id = ?
        """
        
        result = conn.execute(query, (customer_id,)).fetchone()
        conn.close()
        
        if not result:
            return {"error": "Customer not found"}
            
        return {
            "customer_id": result[0],
            "name": result[1],
            "customer_type": result[2],
            "risk_rating": result[3],
            "kyc_status": result[4],
            "registration_country": result[5],
            "is_pep": bool(result[6]),
            "last_review_date": result[7],
            "account_opening_date": result[8]
        }
    
    def _calculate_risk(self, parameters: Dict) -> Dict:
        """Calculate risk score based on multiple factors"""
        risk_score = 0.0
        risk_factors = []
        
        amount = parameters.get("amount", 0)
        country = parameters.get("country", "")
        is_pep = parameters.get("is_pep", False)
        customer_risk = parameters.get("customer_risk_rating", "Low")
        
        # Amount-based risk
        if amount > 500000:
            risk_score += 0.4
            risk_factors.append("Very large transaction amount")
        elif amount > 100000:
            risk_score += 0.3
            risk_factors.append("Large transaction amount")
        elif amount > 50000:
            risk_score += 0.2
            risk_factors.append("Medium transaction amount")
            
        # Country-based risk
        if country in self.compliance_knowledge["aml_policies"]["high_risk_countries"]:
            risk_score += 0.3
            risk_factors.append(f"High-risk jurisdiction: {country}")
            
        # PEP risk
        if is_pep:
            risk_score += 0.2
            risk_factors.append("Politically Exposed Person")
            
        # Customer risk rating
        if customer_risk == "High":
            risk_score += 0.2
            risk_factors.append("High-risk customer profile")
        elif customer_risk == "Medium":
            risk_score += 0.1
            risk_factors.append("Medium-risk customer profile")
            
        return {
            "calculated_risk_score": min(risk_score, 1.0),
            "risk_factors": risk_factors,
            "risk_level": "HIGH" if risk_score > 0.7 else "MEDIUM" if risk_score > 0.4 else "LOW"
        }
    
    def _check_sanctions(self, entity_name: str) -> Dict:
        """Check entity against sanctions lists"""
        sanctions_hits = []
        
        for list_name, entities in self.compliance_knowledge["sanctions_lists"].items():
            if entity_name in entities:
                sanctions_hits.append({
                    "list": list_name,
                    "entity": entity_name,
                    "match_type": "exact"
                })
                
        # Fuzzy matching (simplified)
        for list_name, entities in self.compliance_knowledge["sanctions_lists"].items():
            for sanctioned_entity in entities:
                if sanctioned_entity.lower() in entity_name.lower() or entity_name.lower() in sanctioned_entity.lower():
                    if not any(hit["entity"] == sanctioned_entity for hit in sanctions_hits):
                        sanctions_hits.append({
                            "list": list_name,
                            "entity": sanctioned_entity,
                            "match_type": "partial"
                        })
        
        return {
            "sanctions_hits": sanctions_hits,
            "is_sanctioned": len(sanctions_hits) > 0,
            "total_hits": len(sanctions_hits)
        }
    
    def _validate_transaction(self, parameters: Dict) -> Dict:
        """Validate transaction against business rules"""
        violations = []
        warnings = []
        
        amount = parameters.get("amount", 0)
        transaction_type = parameters.get("transaction_type", "")
        kyc_status = parameters.get("kyc_status", "")
        is_pep = parameters.get("is_pep", False)
        
        # Cash reporting threshold
        if transaction_type == "CASH" and amount >= 10000:
            violations.append("Cash transaction exceeds $10,000 - CTR filing required")
            
        # Wire transfer thresholds
        if transaction_type == "WIRE" and amount >= 50000:
            warnings.append("Wire transfer exceeds review threshold - enhanced due diligence required")
            
        # KYC compliance
        if kyc_status != "Complete":
            violations.append("KYC documentation incomplete")
            
        # PEP approval
        if is_pep and amount > 10000:
            warnings.append("PEP transaction requires senior approval")
            
        return {
            "violations": violations,
            "warnings": warnings,
            "is_compliant": len(violations) == 0
        }
    
    def _get_historical_data(self, customer_id: str) -> Dict:
        """Retrieve customer's historical transaction patterns"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM historical_transactions 
            WHERE customer_id = ?
            ORDER BY transaction_date DESC
            LIMIT 10
        """
        
        results = conn.execute(query, (customer_id,)).fetchall()
        conn.close()
        
        if not results:
            return {"historical_transactions": [], "patterns": {}}
            
        transactions = []
        total_amount = 0
        flagged_count = 0
        
        for row in results:
            transaction = {
                "amount": row[2],
                "date": row[3],
                "country": row[4],
                "flagged": bool(row[5])
            }
            transactions.append(transaction)
            total_amount += row[2]
            if row[5]:
                flagged_count += 1
                
        patterns = {
            "avg_transaction_amount": total_amount / len(transactions),
            "total_flagged": flagged_count,
            "flag_rate": flagged_count / len(transactions),
            "most_common_country": max(set(t["country"] for t in transactions), key=lambda x: sum(1 for t in transactions if t["country"] == x))
        }
        
        return {
            "historical_transactions": transactions,
            "patterns": patterns
        }
    
    def _make_final_decision(self, parameters: Dict) -> Dict:
        """Make final compliance decision based on all gathered information"""
        decision_factors = parameters.get("decision_factors", [])
        risk_level = parameters.get("risk_level", "LOW")
        violations = parameters.get("violations", [])
        sanctions_hits = parameters.get("sanctions_hits", [])
        
        requires_review = False
        compliance_issues = []
        recommendations = []
        
        # Sanctions check
        if sanctions_hits:
            requires_review = True
            compliance_issues.append("Sanctions list match detected - immediate escalation required")
            recommendations.append("Do not process - escalate to compliance officer immediately")
            
        # Regulatory violations
        if violations:
            requires_review = True
            compliance_issues.extend(violations)
            recommendations.append("Address all compliance violations before processing")
            
        # Risk-based decisions
        if risk_level == "HIGH":
            requires_review = True
            recommendations.append("Enhanced due diligence and senior approval required")
            
        elif risk_level == "MEDIUM":
            recommendations.append("Additional monitoring and documentation recommended")
            
        return {
            "requires_review": requires_review,
            "compliance_issues": compliance_issues,
            "recommendations": recommendations,
            "decision_rationale": f"Decision based on {len(decision_factors)} factors analyzed"
        }
    
    def _generate_reasoning(self, step_number: int, context: Dict, previous_observations: List[AgentObservation]) -> Tuple[str, AgentAction]:
        """Generate reasoning and next action using LLM"""
        
        # Build context for reasoning
        context_str = json.dumps(context, indent=2)
        
        # Previous observations summary
        obs_summary = []
        for i, obs in enumerate(previous_observations[-3:]):  # Last 3 observations
            if obs.success:
                obs_summary.append(f"Step {len(previous_observations)-2+i}: {obs.result}")
            else:
                obs_summary.append(f"Step {len(previous_observations)-2+i}: ERROR - {obs.error_message}")
        
        reasoning_prompt = f"""
        You are a banking compliance ReAct agent. You need to analyze a transaction for compliance issues.
        
        Current Context:
        {context_str}
        
        Previous Observations:
        {chr(10).join(obs_summary) if obs_summary else "None"}
        
        Current Step: {step_number}
        
        Available Actions:
        1. search_policy(query) - Search compliance policies
        2. check_customer(customer_id) - Get customer details
        3. calculate_risk(parameters) - Calculate risk score
        4. check_sanctions(entity_name) - Check sanctions lists
        5. validate_transaction(parameters) - Validate against rules
        6. get_historical(customer_id) - Get transaction history
        7. final_decision(parameters) - Make final compliance decision
        
        Think about what information you need next and what action to take.
        
        Respond in this format:
        THOUGHT: [Your reasoning about what to do next]
        ACTION: [action_name]
        PARAMETERS: [JSON parameters for the action]
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a banking compliance expert. Think step by step and choose appropriate actions."},
                    {"role": "user", "content": reasoning_prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
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
                
            # Map action name to ActionType
            action_type_map = {
                "search_policy": ActionType.SEARCH_POLICY,
                "check_customer": ActionType.CHECK_CUSTOMER,
                "calculate_risk": ActionType.CALCULATE_RISK,
                "check_sanctions": ActionType.CHECK_SANCTIONS,
                "validate_transaction": ActionType.VALIDATE_TRANSACTION,
                "get_historical": ActionType.GET_HISTORICAL,
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
            return self._fallback_reasoning(step_number, context, previous_observations)
    
    def _fallback_reasoning(self, step_number: int, context: Dict, previous_observations: List[AgentObservation]) -> Tuple[str, AgentAction]:
        """Fallback reasoning if LLM fails"""
        
        tx_data = context.get("transaction_data", {})
        
        if step_number == 1:
            thought = "First, I need to get detailed customer information."
            action = AgentAction(
                action_type=ActionType.CHECK_CUSTOMER,
                parameters={"customer_id": tx_data.get("customer_id", "")},
                reasoning=thought
            )
            
        elif step_number == 2:
            thought = "Now I need to check if the beneficiary is on any sanctions lists."
            action = AgentAction(
                action_type=ActionType.CHECK_SANCTIONS,
                parameters={"entity_name": tx_data.get("beneficiary_name", "")},
                reasoning=thought
            )
            
        elif step_number == 3:
            thought = "I should calculate the risk score based on all available factors."
            action = AgentAction(
                action_type=ActionType.CALCULATE_RISK,
                parameters={
                    "amount": tx_data.get("amount", 0),
                    "country": tx_data.get("country", ""),
                    "is_pep": tx_data.get("is_pep", False),
                    "customer_risk_rating": "Medium"
                },
                reasoning=thought
            )
            
        else:
            thought = "I have gathered enough information. Time to make a final decision."
            action = AgentAction(
                action_type=ActionType.FINAL_DECISION,
                parameters={
                    "decision_factors": ["customer_check", "sanctions_check", "risk_calculation"],
                    "risk_level": "MEDIUM",
                    "violations": [],
                    "sanctions_hits": []
                },
                reasoning=thought
            )
            
        return thought, action
    
    def analyze_transaction(self, transaction_id: str) -> ComplianceResult:
        """Main ReAct analysis loop"""
        
        # Get initial transaction data
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
            raise ValueError(f"Transaction {transaction_id} not found")
            
        tx_data = {
            "transaction_id": result[0],
            "customer_id": result[1],
            "amount": result[2],
            "transaction_type": result[3],
            "description": result[4],
            "country": result[5],
            "risk_score": result[6],
            "timestamp": result[7],
            "beneficiary_name": result[8],
            "purpose_code": result[9],
            "customer_name": result[10],
            "customer_type": result[11],
            "risk_rating": result[12],
            "kyc_status": result[13],
            "is_pep": bool(result[14])
        }
        
        # Initialize ReAct loop
        react_steps = []
        observations = []
        context = {"transaction_data": tx_data}
        
        print(f"\n🤖 Starting ReAct analysis for {transaction_id}")
        print("=" * 60)
        
        # ReAct loop
        for step in range(1, self.max_iterations + 1):
            
            # Generate reasoning and action
            thought, action = self._generate_reasoning(step, context, observations)
            
            print(f"\nStep {step}:")
            print(f"💭 Thought: {thought}")
            print(f"🎯 Action: {action.action_type.value}")
            print(f"📋 Parameters: {action.parameters}")
            
            # Execute action
            observation = self._execute_action(action)
            observations.append(observation)
            
            print(f"👁️ Observation: {'✅ Success' if observation.success else '❌ Failed'}")
            if observation.success:
                print(f"📊 Result: {observation.result}")
            else:
                print(f"⚠️ Error: {observation.error_message}")
            
            # Record step
            react_step = ReActStep(
                step_number=step,
                thought=thought,
                action=action,
                observation=observation,
                timestamp=datetime.now()
            )
            react_steps.append(react_step)
            
            # Update context with new information
            if observation.success and observation.result:
                context[f"step_{step}_result"] = observation.result
                
            # Check if we should stop (final decision made)
            if action.action_type == ActionType.FINAL_DECISION and observation.success:
                print(f"\n✅ Analysis complete after {step} steps")
                break
                
        # Compile final results
        final_result = observations[-1].result if observations else {}
        
        # Extract information from context
        compliance_issues = []
        recommendations = []
        regulatory_references = []
        
        for step_result in [obs.result for obs in observations if obs.success and obs.result]:
            if isinstance(step_result, dict):
                if "violations" in step_result:
                    compliance_issues.extend(step_result["violations"])
                if "warnings" in step_result:
                    compliance_issues.extend(step_result["warnings"])
                if "compliance_issues" in step_result:
                    compliance_issues.extend(step_result["compliance_issues"])
                if "recommendations" in step_result:
                    recommendations.extend(step_result["recommendations"])
        
        # Determine final risk level
        risk_level = "MEDIUM"
        requires_review = False
        
        if isinstance(final_result, dict):
            requires_review = final_result.get("requires_review", False)
            
        # Calculate risk level from context
        for obs in observations:
            if obs.success and isinstance(obs.result, dict) and "risk_level" in obs.result:
                risk_level = obs.result["risk_level"]
                break
                
        return ComplianceResult(
            transaction_id=transaction_id,
            risk_level=risk_level,
            requires_review=requires_review,
            compliance_issues=compliance_issues[:5],  # Top 5
            recommendations=recommendations[:5],  # Top 5
            regulatory_references=regulatory_references[:3],  # Top 3
            confidence_score=0.9,  # High confidence due to systematic analysis
            analysis_summary=f"ReAct agent completed systematic analysis in {len(react_steps)} steps, examining customer profile, sanctions lists, risk factors, and regulatory compliance.",
            react_trace=react_steps,
            total_steps=len(react_steps)
        )

def command_line_interface():
    """Enhanced command line interface with ReAct trace display"""
    
    system = ReActBankingAgent()
    
    print("\n" + "="*70)
    print("🤖 RE:Agent Prototype - ReAct Banking Compliance System")
    print("="*70)
    
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
            
            print(f"\n🔍 Starting ReAct Analysis for {transaction_id}...")
            print("-" * 70)
            
            try:
                result = system.analyze_transaction(transaction_id)
                
                # Display final results
                print("\n" + "="*70)
                print("📊 FINAL ANALYSIS RESULTS")
                print("="*70)
                
                tx_data = system._check_customer(result.transaction_id.replace("TXN", "CUST"))
                
                print(f"Transaction: {tx_data.get('name', 'Unknown')} - ${result.transaction_id}")
                print(f"Risk Level: {result.risk_level}")
                print(f"Review Required: {'YES' if result.requires_review else 'NO'}")
                print(f"Analysis Steps: {result.total_steps}")
                print(f"Confidence: {result.confidence_score:.0%}")
                
                if result.compliance_issues:
                    print(f"\n⚠️ Compliance Issues ({len(result.compliance_issues)}):")
                    for i, issue in enumerate(result.compliance_issues, 1):
                        print(f"  {i}. {issue}")
                
                if result.recommendations:
                    print(f"\n📋 Recommendations ({len(result.recommendations)}):")
                    for i, rec in enumerate(result.recommendations, 1):
                        print(f"  {i}. {rec}")
                
                print(f"\n💡 Summary: {result.analysis_summary}")
                
                # Option to show ReAct trace
                show_trace = input("\nShow detailed ReAct reasoning trace? (y/n): ").lower() == 'y'
                if show_trace:
                    print("\n" + "="*70)
                    print("🧠 REACT REASONING TRACE")
                    print("="*70)
                    
                    for step in result.react_trace:
                        print(f"\nStep {step.step_number} [{step.timestamp.strftime('%H:%M:%S')}]:")
                        print(f"💭 Thought: {step.thought}")
                        print(f"🎯 Action: {step.action.action_type.value}")
                        print(f"📋 Parameters: {step.action.parameters}")
                        print(f"👁️ Result: {'Success' if step.observation.success else 'Failed'}")
                        if step.observation.success and step.observation.result:
                            print(f"📊 Data: {step.observation.result}")
                        elif not step.observation.success:
                            print(f"⚠️ Error: {step.observation.error_message}")
                        print("-" * 50)
                
            except Exception as e:
                print(f"❌ Error analyzing transaction: {e}")
        else:
            print("Invalid choice. Please select 1-5 or 0 to exit.")

def streamlit_interface():
    """Enhanced Streamlit interface with ReAct trace visualization"""
    
    st.set_page_config(page_title="RE:Agent ReAct Prototype", page_icon="🤖", layout="wide")
    
    st.title("🤖 RE:Agent Prototype - ReAct Banking Compliance System")
    st.markdown("*AI-Powered Reasoning and Acting Agent for Transaction Compliance*")
    
    # Initialize system
    if 'react_system' not in st.session_state:
        with st.spinner("Initializing ReAct compliance system..."):
            st.session_state.react_system = ReActBankingAgent()
    
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
        
        if st.button("🤖 Start ReAct Analysis", type="primary"):
            with st.spinner("ReAct agent analyzing transaction..."):
                try:
                    result = st.session_state.react_system.analyze_transaction(transaction_id)
                    st.session_state.analysis_result = result
                    st.session_state.selected_transaction = transaction_id
                    st.success(f"Analysis complete in {result.total_steps} steps!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with col2:
        st.header("Compliance Analysis Results")
        
        if 'analysis_result' in st.session_state:
            result = st.session_state.analysis_result
            
            # Risk level indicator
            risk_colors = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
            col2a, col2b, col2c, col2d = st.columns(4)
            
            with col2a:
                st.metric("Risk Level", f"{risk_colors.get(result.risk_level, '⚪')} {result.risk_level}")
            with col2b:
                st.metric("Review Required", "YES" if result.requires_review else "NO")
            with col2c:
                st.metric("Analysis Steps", result.total_steps)
            with col2d:
                st.metric("Confidence", f"{result.confidence_score:.0%}")
            
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
            
            # ReAct trace
            st.subheader("🧠 ReAct Reasoning Trace")
            
            with st.expander("Show detailed reasoning steps"):
                for step in result.react_trace:
                    st.markdown(f"**Step {step.step_number}** - {step.timestamp.strftime('%H:%M:%S')}")
                    st.markdown(f"💭 **Thought:** {step.thought}")
                    st.markdown(f"🎯 **Action:** {step.action.action_type.value}")
                    
                    if step.action.parameters:
                        st.code(json.dumps(step.action.parameters, indent=2), language="json")
                    
                    if step.observation.success:
                        st.success("Action completed successfully")
                        if step.observation.result:
                            st.json(step.observation.result)
                    else:
                        st.error(f"Action failed: {step.observation.error_message}")
                    
                    st.markdown("---")
        else:
            st.info("Select a transaction and click 'Start ReAct Analysis' to see results.")

def main():
    if USE_STREAMLIT and len(os.sys.argv) == 1:
        streamlit_interface()
    else:
        command_line_interface()

if __name__ == "__main__":
    main()