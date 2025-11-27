"""
Civic GPS: Complete Policy Navigator with UI, Readiness Score & Intervention Optimizer
"""

import os
import json
import asyncio
import sqlite3
import streamlit as st
from datetime import datetime
from itertools import combinations
from cerebras.cloud.sdk import Cerebras

# ============================================
# CEREBRAS CLIENT SETUP
# ============================================

def get_cerebras_client():
    """Initialize and return Cerebras client"""
    api_key = "csk-24r3yw43vv9cv9f4j4wfjpje9xj9e382rvc384hrr82wtjck"
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY environment variable not set")
    
    return Cerebras(api_key=api_key)

# ============================================
# DATABASE SETUP
# ============================================

class PolicyDatabase:
    """Lightweight SQLite database for storing policy analyses"""
    
    def __init__(self, db_path="policy_analyses.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Policies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS policies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_name TEXT NOT NULL,
                policy_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id INTEGER,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                leverage_score REAL,
                fragility_score REAL,
                readiness_score REAL,
                FOREIGN KEY (policy_id) REFERENCES policies (id)
            )
        ''')
        
        # Variables table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS variables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                name TEXT NOT NULL,
                current_state TEXT,
                value_range TEXT,
                impact_score REAL,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        # Scenarios table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                name TEXT NOT NULL,
                description TEXT,
                base_probability REAL,
                adjusted_probability REAL,
                world_state TEXT,
                key_indicator TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        # Interventions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                name TEXT NOT NULL,
                description TEXT,
                impact TEXT,
                timeline TEXT,
                stakeholders TEXT,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_analysis(self, policy_name, policy_text, results):
        """Save complete analysis to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Save policy
            cursor.execute(
                'INSERT INTO policies (policy_name, policy_text) VALUES (?, ?)',
                (policy_name, policy_text)
            )
            policy_id = cursor.lastrowid
            
            # Save analysis with metadata
            metadata = results.get('reasoning_metadata', {})
            cursor.execute(
                'INSERT INTO analyses (policy_id, leverage_score, fragility_score, readiness_score) VALUES (?, ?, ?, ?)',
                (policy_id, metadata.get('leverage_score', 0), metadata.get('fragility_score', 0), metadata.get('readiness_score', 50))
            )
            analysis_id = cursor.lastrowid
            
            # Save variables
            for var in results.get('all_variables', []):
                cursor.execute(
                    'INSERT INTO variables (analysis_id, name, current_state, value_range, impact_score) VALUES (?, ?, ?, ?, ?)',
                    (analysis_id, var['name'], var['current'], var['range'], var.get('impact_score', 0))
                )
            
            # Save scenarios
            for scenario in results.get('scenarios', []):
                cursor.execute(
                    '''INSERT INTO scenarios 
                    (analysis_id, name, description, base_probability, adjusted_probability, world_state, key_indicator) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (analysis_id, scenario['name'], scenario['description'], 
                     scenario.get('base_probability', scenario['probability']),
                     scenario['probability'], scenario.get('world_state', ''), 
                     scenario['key_indicator'])
                )
            
            # Save interventions
            for intervention in results.get('interventions', []):
                cursor.execute(
                    '''INSERT INTO interventions 
                    (analysis_id, name, description, impact, timeline, stakeholders) 
                    VALUES (?, ?, ?, ?, ?, ?)''',
                    (analysis_id, intervention['name'], intervention['description'],
                     intervention['impact'], intervention['timeline'], intervention['stakeholders'])
                )
            
            conn.commit()
            return analysis_id
            
        except Exception as e:
            print(f"⚠️ Database error: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

# ============================================
# REASONING ENGINE
# ============================================

class PolicyReasoningEngine:
    """Advanced reasoning with Bayesian refinement and modal logic"""
    
    # Interaction matrix for leverage pairs
    INTERACTION_MATRIX = {
        ("high", "high"): 1.8,    # Strong synergy
        ("high", "medium"): 1.4,
        ("high", "low"): 0.6,     # Counteracting
        ("medium", "high"): 1.4,
        ("medium", "medium"): 1.0, # Neutral
        ("medium", "low"): 0.7,
        ("low", "high"): 0.6,
        ("low", "medium"): 0.7,
        ("low", "low"): 0.4       # Strong negative synergy
    }
    
    @staticmethod
    def enhanced_normalize_state(state):
        """Enhanced state normalization with better keyword matching"""
        if state is None:
            return "medium"
            
        state_lower = str(state).lower()
        
        # High state indicators
        high_indicators = ['high', 'max', 'fast', 'strong', 'extensive', 'excellent', 
                          'optimal', 'complete', 'approved', 'full', 'rapid', 'increased']
        # Low state indicators  
        low_indicators = ['low', 'min', 'slow', 'weak', 'limited', 'poor', 'minimal',
                         'pending', 'delayed', 'rejected', 'partial', 'decreased', 'short']
        
        high_count = sum(1 for indicator in high_indicators if indicator in state_lower)
        low_count = sum(1 for indicator in low_indicators if indicator in state_lower)
        
        if high_count > low_count:
            return "high"
        elif low_count > high_count:
            return "low"
        else:
            # Check for numeric values
            if any(char.isdigit() for char in state_lower):
                # Try to parse numbers for relative positioning
                numbers = [int(s) for s in state_lower.split() if s.isdigit()]
                if numbers:
                    avg_num = sum(numbers) / len(numbers)
                    if avg_num > 50:  # Arbitrary threshold
                        return "high"
                    elif avg_num < 20:
                        return "low"
            return "medium"
    
    @staticmethod
    def calculate_leverage_score(var1, var2):
        """Calculate leverage score using enhanced interaction matrix"""
        state1 = PolicyReasoningEngine.enhanced_normalize_state(var1['current'])
        state2 = PolicyReasoningEngine.enhanced_normalize_state(var2['current'])
        
        base_score = PolicyReasoningEngine.INTERACTION_MATRIX.get(
            (state1, state2), 1.0
        )
        
        # Add range breadth bonus
        range_bonus = 0
        for var in [var1, var2]:
            range_text = var.get('range', '').lower()
            if 'low' in range_text and 'high' in range_text:
                range_bonus += 0.2
            if any(word in range_text for word in ['wide', 'broad', 'extensive']):
                range_bonus += 0.1
        
        return min(2.0, base_score + range_bonus)  # Cap at 2.0
    
    @staticmethod
    def bayesian_refine_probabilities(scenarios, leverage_score):
        """Apply meaningful Bayesian adjustment to scenario probabilities"""
        # Store base probabilities
        base_probs = [scenario.get('base_probability', scenario['probability']) 
                     for scenario in scenarios]
        
        # Apply leverage-based adjustment
        # Higher leverage score increases optimistic scenario probability
        adjustment_factors = []
        for i, scenario in enumerate(scenarios):
            world_state = scenario.get('world_state', '')
            if world_state == 'optimistic':
                # Optimistic scenarios benefit from high leverage
                adjustment_factors.append(leverage_score)
            elif world_state == 'risk':
                # Risk scenarios are penalized by high leverage
                adjustment_factors.append(2.0 - leverage_score)  # Inverse
            else:
                # Expected scenarios get moderate adjustment
                adjustment_factors.append(1.0)
        
        # Apply adjustments
        adjusted_probs = [base_probs[i] * adjustment_factors[i] 
                         for i in range(len(scenarios))]
        
        # Normalize to sum to 100%
        total = sum(adjusted_probs)
        if total > 0:
            adjusted_probs = [max(5, min(70, round((p / total) * 100))) for p in adjusted_probs]
        else:
            adjusted_probs = [33, 33, 34]
        
        # Ensure sum is exactly 100 with small adjustments
        diff = 100 - sum(adjusted_probs)
        if diff != 0:
            # Distribute difference to non-extreme probabilities
            for i in range(abs(diff)):
                idx = i % len(adjusted_probs)
                if 10 < adjusted_probs[idx] < 70:  # Avoid extremes
                    adjusted_probs[idx] += 1 if diff > 0 else -1
        
        return adjusted_probs
    
    @staticmethod
    def create_possible_worlds(leverage_pair):
        """Create enhanced possible worlds map"""
        v1, v2 = leverage_pair
        
        v1_state = PolicyReasoningEngine.enhanced_normalize_state(v1['current'])
        v2_state = PolicyReasoningEngine.enhanced_normalize_state(v2['current'])
        
        return {
            "optimistic": {
                v1['name']: "high",
                v2['name']: "high",
                "description": f"Optimal conditions with {v1['name']} and {v2['name']} both favorable"
            },
            "expected": {
                v1['name']: v1_state,
                v2['name']: v2_state, 
                "description": f"Current trajectory with {v1['name']} at {v1_state} and {v2['name']} at {v2_state}"
            },
            "risk": {
                v1['name']: "low",
                v2['name']: "low",
                "description": f"Challenging conditions with both {v1['name']} and {v2['name']} unfavorable"
            }
        }
    
    @staticmethod
    def calculate_fragility(worlds):
        """Calculate realistic policy fragility score"""
        expected_world = worlds["expected"]
        risk_world = worlds["risk"]
        
        if not expected_world or not risk_world:
            return 5.0  # Default moderate fragility
        
        # Calculate state transition distances
        transitions = 0
        total_possible = 0
        
        for var in expected_world:
            if var in risk_world:
                total_possible += 1
                expected_state = expected_world[var]
                risk_state = risk_world[var]
                
                # Score transition difficulty
                if expected_state == "high" and risk_state == "low":
                    transitions += 2  # Major transition
                elif expected_state != risk_state:
                    transitions += 1  # Minor transition
        
        if total_possible == 0:
            return 5.0
        
        # Normalize to 0-10 scale with more realistic distribution
        fragility = (transitions / (total_possible * 2)) * 10
        return round(min(9.5, fragility), 1)  # Cap below 10 for realism

# ============================================
# TEST POLICIES
# ============================================

TEST_POLICIES = {
    "climate": {
        "name": "Climate Adaptation Fund",
        "text": """
        Proposed Climate Adaptation Fund: 
        Allocate $50M over 3 years for coastal resilience projects in vulnerable communities.
        Requires community input, regulatory approval from environmental agencies, and 25% matching funds from local governments.
        Success depends on stakeholder buy-in, rapid deployment before next storm season, and maintaining public support.
        Key risks include bureaucratic delays, funding shortfalls, and competing priorities at local level.
        Expected outcomes: protect 100,000 residents, create 500 green jobs, and prevent $200M in climate damages.
        """
    },
    "housing": {
        "name": "Affordable Housing Initiative", 
        "text": """
        Affordable Housing Expansion Act:
        Provide tax incentives and streamlined permitting for developers who allocate 20% of new units as affordable housing.
        Includes $10M for infrastructure upgrades in designated development zones.
        Requires coordination between city planning, housing authorities, and community boards.
        Success factors: developer participation rates, construction timelines, neighborhood acceptance.
        Targets: 5,000 new affordable units over 5 years, reduce homelessness by 15%.
        """
    },
    "education": {
        "name": "Digital Literacy Program",
        "text": """
        Community Digital Literacy Program:
        Deploy technology centers in 50 underserved communities with broadband access and digital skills training.
        Budget: $15M annually for equipment, instructors, and maintenance.
        Partnerships required with local schools, libraries, and tech companies for volunteers.
        Key metrics: participant completion rates, job placement outcomes, technology adoption.
        Goals: train 25,000 individuals annually, bridge digital divide in rural and low-income areas.
        """
    }
}

# ============================================
# PROMPT TEMPLATES
# ============================================

EXTRACT_VARIABLES_PROMPT = """You are analyzing a policy document to extract key variables that influence policy outcomes.

Policy Text:
{policy_text}

Extract 5-8 key variables that could significantly impact this policy's success or failure.
For each variable, provide:
1. Variable name (2-4 words, clear and specific)
2. Current state (brief description of current status)
3. Potential range (low/medium/high or specific numeric range if applicable)

Focus on variables that have high impact potential and variability in outcomes.

Return your response as a JSON object with a "variables" array containing the extracted variables."""

GENERATE_SCENARIOS_PROMPT = """You are a policy analyst creating future scenarios based on key leverage variables.

Policy Summary: {policy_summary}

Analysis identified these TWO CRITICAL VARIABLES that most influence outcomes:
1. {var1} (Current: {current1}, Range: {range1})
2. {var2} (Current: {current2}, Range: {range2})

Generate exactly 3 distinct future scenarios based on different combinations of these variables:

SCENARIO A (Optimistic): Both variables develop favorably
SCENARIO B (Expected): Variables follow current trends with moderate outcomes  
SCENARIO C (Risk): Variables work against policy goals

For each scenario provide:
- Name (3-5 words that capture the scenario essence)
- Description (2-3 sentences explaining the scenario dynamics)
- Probability (percentage likelihood, must sum to 100% across all scenarios)
- Key indicator to monitor (specific metric that signals this scenario emerging)

Ensure probabilities sum to 100% and scenarios are distinct and realistic.

Return your response as a JSON object with a "scenarios" array containing the three scenarios."""

INTERVENTION_PROMPT = """As a policy optimization expert, suggest targeted interventions for maximum impact.

Policy: {policy_name}
Policy Summary: {policy_summary}

Critical Leverage Variables:
1. {var1} (Current: {current1})
2. {var2} (Current: {current2})

Identified Scenarios:
{scenarios_summary}

Based on this analysis, suggest exactly 3 high-impact interventions that would:
- Improve the optimistic scenario probability
- Mitigate risks in the challenging scenario  
- Address the leverage variables directly

For each intervention, provide:
- Intervention name (3-5 words)
- Description (what specific actions to take)
- Expected impact (low/medium/high)
- Timeline (short/medium/long-term)
- Key stakeholders involved

Return your response as a JSON object with an "interventions" array containing the three interventions."""

# ============================================
# JSON SCHEMAS
# ============================================

VARIABLES_SCHEMA = {
    "type": "object",
    "properties": {
        "variables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "current": {"type": "string"},
                    "range": {"type": "string"}
                },
                "required": ["name", "current", "range"],
                "additionalProperties": False
            }
        }
    },
    "required": ["variables"],
    "additionalProperties": False
}

SCENARIOS_SCHEMA = {
    "type": "object",
    "properties": {
        "scenarios": {
            "type": "array", 
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "probability": {"type": "integer"},
                    "key_indicator": {"type": "string"}
                },
                "required": ["name", "description", "probability", "key_indicator"],
                "additionalProperties": False
            }
        }
    },
    "required": ["scenarios"],
    "additionalProperties": False
}

INTERVENTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "interventions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "impact": {"type": "string"},
                    "timeline": {"type": "string"},
                    "stakeholders": {"type": "string"}
                },
                "required": ["name", "description", "impact", "timeline", "stakeholders"],
                "additionalProperties": False
            }
        }
    },
    "required": ["interventions"],
    "additionalProperties": False
}

# ============================================
# ENHANCED CIVIC GPS WITH NEW FEATURES
# ============================================

class CivicGPS:
    """Complete policy navigation with readiness scores and interventions"""
    
    def __init__(self, model="llama-3.3-70b"):
        self.client = get_cerebras_client()
        self.model = model
        self.db = PolicyDatabase()
        self.reasoning_engine = PolicyReasoningEngine()
        self.variables = []
        self.leverage_pair = None
        self.scenarios = []
        self.interventions = []
        self.analysis_metadata = {}
    
    async def analyze_policy(self, policy_text, policy_name="Custom Policy"):
        """Complete analysis pipeline with interventions"""
        
        print("📊 Extracting key variables...")
        self.variables = await self._extract_variables(policy_text)
        
        print("🎯 Finding leverage pair with interaction scoring...")
        self.leverage_pair = self._find_leverage_pair()
        
        # Calculate REAL leverage score
        leverage_score = self.reasoning_engine.calculate_leverage_score(
            self.leverage_pair[0], self.leverage_pair[1]
        )
        self.analysis_metadata['leverage_score'] = leverage_score
        
        print("🔮 Generating future scenarios...")
        raw_scenarios = await self._generate_scenarios(policy_text)
        
        print("📈 Applying Bayesian probability refinement...")
        self.scenarios = self._refine_scenarios(raw_scenarios, leverage_score)
        
        print("🌐 Creating modal possible worlds...")
        worlds = self.reasoning_engine.create_possible_worlds(self.leverage_pair)
        fragility_score = self.reasoning_engine.calculate_fragility(worlds)
        self.analysis_metadata['fragility_score'] = fragility_score
        self.analysis_metadata['possible_worlds'] = worlds
        
        print("🎯 Generating intervention strategies...")
        self.interventions = await self._generate_interventions(policy_text, policy_name)
        
        print("📊 Calculating policy readiness score...")
        readiness_score = self._calculate_readiness_score()
        self.analysis_metadata['readiness_score'] = readiness_score
        
        print("💾 Saving analysis to database...")
        results = self.get_results()
        analysis_id = self.db.save_analysis(policy_name, policy_text, results)
        self.analysis_metadata['analysis_id'] = analysis_id
        
        return results
    
    def _calculate_readiness_score(self):
        """Calculate Policy Readiness Index (0-100)"""
        base_score = 50  # Neutral starting point
        
        # Leverage alignment bonus
        leverage_score = self.analysis_metadata.get('leverage_score', 1.0)
        alignment_bonus = (leverage_score - 1.0) * 25  # Convert to -15 to +20 range
        
        # Risk penalty based on worst-case scenario probability
        risk_prob = max(scenario['probability'] for scenario in self.scenarios 
                       if scenario.get('world_state') == 'risk')
        risk_penalty = (risk_prob / 100) * 20  # Up to 20 point penalty
        
        # Variable state bonus
        positive_vars = sum(1 for var in self.variables 
                           if self.reasoning_engine.enhanced_normalize_state(var['current']) == 'high')
        variable_bonus = min(10, positive_vars * 2)  # Up to 10 point bonus
        
        readiness = base_score + alignment_bonus - risk_penalty + variable_bonus
        return max(0, min(100, int(readiness)))
    
    async def _generate_interventions(self, policy_text, policy_name):
        """Generate targeted interventions using Cerebras"""
        v1, v2 = self.leverage_pair
        
        # Create scenarios summary
        scenarios_summary = "\n".join([
            f"- {s['name']} ({s['probability']}%): {s['description'][:100]}..."
            for s in self.scenarios
        ])
        
        prompt = INTERVENTION_PROMPT.format(
            policy_name=policy_name,
            policy_summary=policy_text[:500],
            var1=v1['name'], current1=v1['current'],
            var2=v2['name'], current2=v2['current'],
            scenarios_summary=scenarios_summary
        )
        
        result = await self._call_cerebras(prompt, INTERVENTIONS_SCHEMA, temperature=0.7)
        
        if result and 'interventions' in result and len(result['interventions']) == 3:
            return result['interventions']
        
        # Fallback interventions
        return [
            {
                "name": f"Enhance {v1['name']}",
                "description": f"Implement targeted programs to improve {v1['name']} through strategic partnerships and resource allocation.",
                "impact": "high",
                "timeline": "medium-term",
                "stakeholders": f"Policy leads, {v1['name']} stakeholders"
            },
            {
                "name": f"Strengthen {v2['name']}",
                "description": f"Develop comprehensive strategy to bolster {v2['name']} with monitoring and feedback mechanisms.",
                "impact": "high", 
                "timeline": "short-term",
                "stakeholders": f"Implementation team, {v2['name']} partners"
            },
            {
                "name": "Risk Mitigation Framework",
                "description": "Create early warning system and contingency plans for scenario alignment shifts.",
                "impact": "medium",
                "timeline": "long-term",
                "stakeholders": "Risk management, all stakeholders"
            }
        ]
    
    def _refine_scenarios(self, scenarios, leverage_score):
        """Fixed scenario refinement with actual probability changes"""
        # Store base probabilities
        for scenario in scenarios:
            scenario['base_probability'] = scenario['probability']
        
        # Apply meaningful Bayesian adjustment
        adjusted_probs = self.reasoning_engine.bayesian_refine_probabilities(
            scenarios, leverage_score
        )
        
        # Update scenarios with refined probabilities
        for i, scenario in enumerate(scenarios):
            scenario['probability'] = adjusted_probs[i]
            # Add world state classification
            if i == 0:
                scenario['world_state'] = 'optimistic'
            elif i == 1:
                scenario['world_state'] = 'expected' 
            else:
                scenario['world_state'] = 'risk'
        
        return scenarios
    
    async def _extract_variables(self, policy_text):
        prompt = EXTRACT_VARIABLES_PROMPT.format(policy_text=policy_text[:2000])
        result = await self._call_cerebras(prompt, VARIABLES_SCHEMA)
        
        if result and 'variables' in result and len(result['variables']) >= 3:
            return result['variables'][:8]
        
        return [
            {"name": "implementation_speed", "current": "slow", "range": "slow to fast"},
            {"name": "stakeholder_alignment", "current": "medium", "range": "low to high"},
            {"name": "budget_allocation", "current": "$2M", "range": "$1M to $5M"},
            {"name": "regulatory_approval", "current": "pending", "range": "rejected to approved"},
            {"name": "public_support", "current": "moderate", "range": "low to high"}
        ]
    
    def _find_leverage_pair(self):
        if len(self.variables) < 2:
            return (self.variables[0], self.variables[0])
        
        def enhanced_score_variable(var):
            score = 0
            range_text = var.get('range', '').lower()
            name_text = var.get('name', '').lower()
            
            if 'low' in range_text and 'high' in range_text:
                score += 3
            elif 'min' in range_text and 'max' in range_text:
                score += 2
            elif 'to' in range_text or '-' in range_text:
                score += 1
            
            impact_weights = {
                'funding': 3, 'budget': 3, 'approval': 3, 'regulatory': 3,
                'support': 2, 'stakeholder': 2, 'public': 2, 'alignment': 2,
                'speed': 2, 'timeline': 2, 'participation': 2, 'engagement': 2
            }
            
            for keyword, weight in impact_weights.items():
                if keyword in name_text:
                    score += weight
                    break
            
            if any(char.isdigit() for char in range_text):
                score += 1
                
            return score
        
        best_pair = None
        best_score = -1
        
        for v1, v2 in combinations(self.variables, 2):
            pair_score = enhanced_score_variable(v1) + enhanced_score_variable(v2)
            
            v1_state = self.reasoning_engine.enhanced_normalize_state(v1['current'])
            v2_state = self.reasoning_engine.enhanced_normalize_state(v2['current'])
            interaction_bonus = self.reasoning_engine.INTERACTION_MATRIX.get(
                (v1_state, v2_state), 1.0
            )
            pair_score *= interaction_bonus
            
            if pair_score > best_score:
                best_score = pair_score
                best_pair = (v1, v2)
        
        return best_pair or (self.variables[0], self.variables[1])
    
    async def _generate_scenarios(self, policy_text):
        v1, v2 = self.leverage_pair
        
        prompt = GENERATE_SCENARIOS_PROMPT.format(
            policy_summary=policy_text[:500],
            var1=v1['name'], current1=v1['current'], range1=v1['range'],
            var2=v2['name'], current2=v2['current'], range2=v2['range']
        )
        
        result = await self._call_cerebras(prompt, SCENARIOS_SCHEMA, temperature=0.7)
        
        if result and 'scenarios' in result and len(result['scenarios']) == 3:
            return result['scenarios']
        
        return [
            {
                "name": "Accelerated Success",
                "description": f"Policy achieves goals ahead of schedule through optimal alignment of {v1['name']} and {v2['name']}.",
                "probability": 30,
                "key_indicator": f"{v1['name']} trending upward"
            },
            {
                "name": "Expected Trajectory", 
                "description": f"Policy follows projected timeline with {v1['name']} and {v2['name']} developing as anticipated.",
                "probability": 50,
                "key_indicator": f"{v2['name']} remains stable"
            },
            {
                "name": "Challenged Implementation",
                "description": f"Policy faces resistance due to misalignment between {v1['name']} and {v2['name']}, requiring intervention.",
                "probability": 20,
                "key_indicator": f"{v1['name']} and {v2['name']} moving in opposite directions"
            }
        ]
    
    async def _call_cerebras(self, prompt, schema, temperature=0.3):
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema", 
                    "json_schema": {
                        "name": "structured_output",
                        "strict": True,
                        "schema": schema
                    }
                },
                temperature=temperature
            )
            
            content = completion.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            print(f"⚠️ Cerebras API error: {e}")
            return None
    
    def get_results(self):
        v1, v2 = self.leverage_pair
        
        return {
            "policy_name": self.analysis_metadata.get('policy_name', 'Unknown Policy'),
            "leverage_pair": {
                "variable_1": {
                    "name": v1['name'],
                    "current": v1['current'],
                    "range": v1['range']
                },
                "variable_2": {
                    "name": v2['name'],
                    "current": v2['current'],
                    "range": v2['range']
                },
                "control_explanation": f"These two factors create the majority of outcome variance through their interaction"
            },
            "scenarios": self.scenarios,
            "interventions": self.interventions,
            "all_variables": self.variables,
            "reasoning_metadata": {
                "leverage_score": self.analysis_metadata.get('leverage_score', 0),
                "fragility_score": self.analysis_metadata.get('fragility_score', 0),
                "readiness_score": self.analysis_metadata.get('readiness_score', 50),
                "possible_worlds": self.analysis_metadata.get('possible_worlds', {}),
                "analysis_id": self.analysis_metadata.get('analysis_id')
            },
            "insight": f"Focus monitoring and intervention on {v1['name']} and {v2['name']} to steer policy outcomes"
        }

# ============================================
# CLEAN STREAMLIT UI
# ============================================

def create_clean_ui():
    """Create a clean, focused Streamlit UI"""
    
    st.set_page_config(
        page_title="Civic GPS: Policy Navigator",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for clean design
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1.1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .scenario-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .intervention-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
        margin-bottom: 1rem;
    }
    .leverage-highlight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        color: white;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .variable-table {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main Header
    st.markdown('<h1 class="main-header">🧭 Civic GPS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-powered policy analysis that identifies critical leverage points and optimal interventions</p>', unsafe_allow_html=True)
    
    # Input Section
    st.markdown("---")
    st.subheader("📝 Policy Input")
    
    input_col1, input_col2 = st.columns([2, 1])
    
    with input_col1:
        # Policy text input
        policy_text = st.text_area(
            "Paste your policy text below:",
            height=200,
            placeholder="Enter policy document text here...\n\nExample: 'Proposed Climate Adaptation Fund: Allocate $50M over 3 years for coastal resilience projects...'",
            help="The policy text should describe the goals, requirements, risks, and expected outcomes."
        )
    
    with input_col2:
        # Quick load buttons
        st.write("**Quick Load Demo Policies:**")
        
        if st.button("🌊 Climate Policy", use_container_width=True):
            policy_text = TEST_POLICIES["climate"]["text"]
            st.session_state.policy_text = policy_text
            st.rerun()
            
        if st.button("🏠 Housing Policy", use_container_width=True):
            policy_text = TEST_POLICIES["housing"]["text"]
            st.session_state.policy_text = policy_text
            st.rerun()
            
        if st.button("💻 Digital Policy", use_container_width=True):
            policy_text = TEST_POLICIES["education"]["text"]
            st.session_state.policy_text = policy_text
            st.rerun()
        
        st.write("---")
        policy_name = st.text_input("Policy Name (optional):", placeholder="e.g., Climate Adaptation Fund")
    
    # Use session state to persist policy text
    if 'policy_text' in st.session_state and not policy_text:
        policy_text = st.session_state.policy_text
    
    # Analyze Button
    analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
    with analyze_col2:
        analyze_button = st.button(
            "🚀 Analyze Policy", 
            type="primary", 
            use_container_width=True,
            disabled=not policy_text.strip()
        )
    
    if not policy_text.strip():
        st.info("👆 Enter policy text or select a demo policy to begin analysis")
    
    # Analysis Results
    if analyze_button and policy_text:
        with st.spinner("🧠 Analyzing policy with advanced AI reasoning..."):
            try:
                gps = CivicGPS()
                policy_display_name = policy_name or "Policy Analysis"
                results = asyncio.run(gps.analyze_policy(policy_text, policy_display_name))
                st.session_state.results = results
                st.session_state.analysis_done = True
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.stop()
    
    # Display Results
    if st.session_state.get('analysis_done') and 'results' in st.session_state:
        results = st.session_state.results
        metadata = results['reasoning_metadata']
        
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        # Key Metrics
        st.write("**Key Metrics:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            readiness = metadata.get('readiness_score', 50)
            st.metric(
                "Policy Readiness", 
                f"{readiness}/100",
                delta=f"{readiness-50}" if readiness != 50 else "0"
            )
        
        with col2:
            st.metric("Leverage Score", f"{metadata.get('leverage_score', 0):.2f}")
        
        with col3:
            st.metric("Fragility", f"{metadata.get('fragility_score', 0)}/10")
        
        with col4:
            optimistic_prob = next((s['probability'] for s in results['scenarios'] if s.get('world_state') == 'optimistic'), 0)
            st.metric("Optimistic Outlook", f"{optimistic_prob}%")
        
        # Leverage Pair
        st.markdown("---")
        st.subheader("🎯 Critical Leverage Variables")
        
        v1, v2 = results['leverage_pair']['variable_1'], results['leverage_pair']['variable_2']
        
        lev_col1, lev_col2, lev_col3 = st.columns([2, 1, 2])
        
        with lev_col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🔍 {v1['name']}</h3>
                <p><strong>Current State:</strong> {v1['current']}</p>
                <p><strong>Potential Range:</strong> {v1['range']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with lev_col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h1>⚡</h1>
                <p><strong>High<br>Interaction</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        with lev_col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🔍 {v2['name']}</h3>
                <p><strong>Current State:</strong> {v2['current']}</p>
                <p><strong>Potential Range:</strong> {v2['range']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.caption("💡 These two variables have the highest impact on policy outcomes")
        
        # Scenarios and Interventions
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔮 Future Scenarios")
            for scenario in results['scenarios']:
                with st.container():
                    st.markdown(f"""
                    <div class='scenario-card'>
                        <h4>{scenario['name']} ({scenario['probability']}%)</h4>
                        <p>{scenario['description']}</p>
                        <div style='background: #f0f2f6; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;'>
                            <small>📈 <strong>Watch:</strong> {scenario['key_indicator']}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🎯 Recommended Interventions")
            for intervention in results.get('interventions', []):
                with st.container():
                    impact_color = {
                        'high': '#ff6b6b',
                        'medium': '#feca57', 
                        'low': '#48dbfb'
                    }.get(intervention['impact'].lower(), '#666')
                    
                    st.markdown(f"""
                    <div class='intervention-card'>
                        <h4>{intervention['name']}</h4>
                        <p>{intervention['description']}</p>
                        <div style='display: flex; justify-content: space-between; margin-top: 0.5rem;'>
                            <small>🕒 <strong>Timeline:</strong> {intervention['timeline']}</small>
                            <small>💥 <strong>Impact:</strong> <span style='color: {impact_color};'>{intervention['impact'].title()}</span></small>
                        </div>
                        <div style='background: #e3f2fd; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;'>
                            <small>👥 <strong>Stakeholders:</strong> {intervention['stakeholders']}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Additional Analysis
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 All Variables")
            variables_data = []
            for var in results['all_variables']:
                variables_data.append({
                    "Variable": var['name'],
                    "Current State": var['current'],
                    "Range": var['range']
                })
            st.dataframe(variables_data, use_container_width=True)
        
        with col2:
            st.subheader("🌐 Possible Worlds")
            worlds = metadata.get('possible_worlds', {})
            for world_name, world_state in worlds.items():
                if isinstance(world_state, dict) and 'description' in world_state:
                    st.write(f"**{world_name.title()} World**")
                    st.caption(world_state['description'])
                    st.write("")
        
        # Export
        st.markdown("---")
        st.subheader("📥 Export Results")
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="Download Full Analysis (JSON)",
            data=json_str,
            file_name=f"policy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    # Sidebar with Instructions
    with st.sidebar:
        st.header("Quick Start Guide")
        
        st.markdown("""
        **1. Input Policy Text**
        - Paste your policy document
        - Or use demo policies for quick testing
        
        **2. Analyze**
        - Click the Analyze Policy button
        - AI processes with advanced reasoning
        
        **3. Review Results**
        - **Leverage Variables**: Most impactful factors
        - **Scenarios**: Future outlook probabilities  
        - **Interventions**: Actionable recommendations
        - **Metrics**: Readiness and risk scores
        
        **4. Take Action**
        - Focus on leverage variables
        - Implement recommended interventions
        - Monitor key indicators
        - Download full analysis
        """)
        
        st.markdown("---")
        st.caption("""
        **Powered by:**
        - Cerebras LLM with structured outputs
        - Bayesian probability refinement
        - Modal logic reasoning
        - Policy optimization algorithms
        """)

# ============================================
# INITIALIZATION & MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Check for API key
    if not os.environ.get("CEREBRAS_API_KEY"):
        st.error("❌ CEREBRAS_API_KEY environment variable not set")
        st.info("💡 Please set your API key: `export CEREBRAS_API_KEY='your-api-key-here'`")
        st.stop()
    
    # Initialize session state
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'policy_text' not in st.session_state:
        st.session_state.policy_text = ""
    
    # Create the clean UI
    create_clean_ui()