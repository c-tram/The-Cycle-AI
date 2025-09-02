"""
Intelligence Integration Layer
Brings together semantic understanding, learning, and contextual reasoning.
This replaces the old rule-based system with true intelligence.
"""

from .semantic_engine import SemanticEngine
from .learning_system import LearningKnowledgeGraph
from .contextual_reasoner import ContextualReasoner
from typing import Dict, List, Optional, Tuple
import json

class IntelligentQuery:
    """Represents a query with full understanding"""
    def __init__(self, original_query: str):
        self.original_query = original_query
        self.semantic_understanding = None
        self.learned_insights = None
        self.contextual_intent = None
        self.execution_plan = None
        self.confidence = 0.0

class IntelligentAgent:
    """
    The brain that coordinates semantic understanding, learning, and reasoning.
    This is your "truly intelligent" system.
    """
    
    def __init__(self):
        self.semantic_engine = SemanticEngine()
        self.knowledge_graph = LearningKnowledgeGraph()
        self.contextual_reasoner = ContextualReasoner()
        
        # Load any previously learned knowledge
        try:
            self.knowledge_graph.load_knowledge("learned_knowledge.json")
        except:
            pass  # Start fresh if no saved knowledge
        
    def process_query(self, query: str, conversation_history: List[str] = None) -> IntelligentQuery:
        """
        Process a query with full intelligence - understanding, learning, and reasoning.
        This is the replacement for your old rule-based parse_query_intent.
        """
        
        intelligent_query = IntelligentQuery(query)
        
        # 1. Semantic Understanding - What does this query mean?
        intelligent_query.semantic_understanding = self.semantic_engine.understand_query(query)
        
        # 2. Apply Learned Knowledge - What have we learned about similar queries?
        intelligent_query.learned_insights = self.knowledge_graph.predict_best_approach(query)
        
        # 3. Contextual Reasoning - Why is the user asking this?
        intelligent_query.contextual_intent = self.contextual_reasoner.understand_intent(query, conversation_history)
        
        # 4. Create Unified Execution Plan
        intelligent_query.execution_plan = self._create_execution_plan(intelligent_query)
        
        # 5. Calculate Overall Confidence
        intelligent_query.confidence = self._calculate_confidence(intelligent_query)
        
        return intelligent_query
    
    def _create_execution_plan(self, iq: IntelligentQuery) -> Dict:
        """Create a unified plan that combines all intelligence sources"""
        
        # Determine the best endpoint based on query content
        endpoint = self._choose_best_endpoint(iq)
        
        plan = {
            "data_source": f"https://thecycle.online/{endpoint}",
            "endpoint": endpoint,
            "search_strategy": "frontend_scraping",
            "table_processing": {
                "expected_columns": [],
                "filters": [],
                "sorting": None
            },
            "response_strategy": {
                "style": "informative",
                "include_confidence": False,
                "followup_suggestions": []
            },
            "reasoning_chain": []
        }
        
        # Incorporate semantic understanding
        if iq.semantic_understanding:
            sem = iq.semantic_understanding
            
            # Use semantic constraints
            for constraint in sem.get("constraints", []):
                plan["table_processing"]["filters"].append({
                    "type": constraint["type"],
                    "column_concept": constraint.get("applies_to", "unknown"),
                    "operator": constraint.get("operator", ">="),
                    "value": constraint["value"]
                })
                column_name = constraint.get("applies_to", "unknown")
                plan["reasoning_chain"].append(f"Semantic: Filter {column_name} {constraint.get('operator', '>=')} {constraint['value']}")
            
            # Use semantic intent for sorting
            intent = sem.get("intent")
            target_concept = None
            concepts_found = sem.get("concepts_found", [])
            if concepts_found:
                target_concept = concepts_found[0]  # Use first concept as target
            
            if intent == "find_maximum":
                plan["table_processing"]["sorting"] = {"direction": "desc", "concept": target_concept}
                plan["reasoning_chain"].append(f"Semantic: Find maximum {target_concept}")
            elif intent == "find_minimum":
                plan["table_processing"]["sorting"] = {"direction": "asc", "concept": target_concept}
                plan["reasoning_chain"].append(f"Semantic: Find minimum {target_concept}")
        
        # Incorporate learned knowledge
        if iq.learned_insights and iq.learned_insights["confidence"] > 0.5:
            learned = iq.learned_insights
            
            # Use learned column mappings
            for col in learned.get("suggested_columns", []):
                plan["table_processing"]["expected_columns"].append(col)
                plan["reasoning_chain"].append(f"Learned: Column '{col}' is reliable for this query type")
            
            # Use learned patterns
            for reason in learned.get("reasoning", []):
                plan["reasoning_chain"].append(f"Experience: {reason}")
        
        # Incorporate contextual reasoning
        if iq.contextual_intent:
            context = iq.contextual_intent
            
            # Adjust response style based on context
            reasoning_strategy = self.contextual_reasoner.generate_reasoning_strategy(context)
            plan["response_strategy"].update(reasoning_strategy)
            
            # Add constraint filters from context
            for constraint_name, value in context.constraints.items():
                if constraint_name == "min_games":
                    plan["table_processing"]["filters"].append({
                        "type": "minimum",
                        "column_concept": "games_played", 
                        "operator": ">=",
                        "value": value
                    })
                    plan["reasoning_chain"].append(f"Context: Minimum {value} games required")
            
            # Set confidence communication
            if context.confidence_threshold > 0.7:
                plan["response_strategy"]["include_confidence"] = True
                plan["reasoning_chain"].append("Context: High confidence required, will include data reliability notes")
        
        return plan
    
    def _choose_best_endpoint(self, iq: IntelligentQuery) -> str:
        """Choose the best endpoint based on query analysis"""
        
        query_lower = iq.original_query.lower()
        
        # Check for pitching-related keywords
        pitching_keywords = [
            'pitch', 'pitching', 'era', 'whip', 'strikeout', 'k/9', 'fip', 'war', 
            'innings', 'ip', 'so', 'bb', 'hitter', 'batter', 'runs allowed', 'ra'
        ]
        
        # Check for batting/hitting-related keywords
        batting_keywords = [
            'hit', 'batting', 'avg', 'obp', 'slg', 'ops', 'hr', 'homerun', 'rbi',
            'runs', 'hits', 'doubles', 'triples', 'steals', 'sb', 'contact', 'power'
        ]
        
        # Count keyword matches
        pitching_score = sum(1 for keyword in pitching_keywords if keyword in query_lower)
        batting_score = sum(1 for keyword in batting_keywords if keyword in query_lower)
        
        # Use semantic understanding if available
        if iq.semantic_understanding:
            concepts = iq.semantic_understanding.get("concepts_found", [])
            for concept in concepts:
                concept_lower = concept.lower()
                if any(word in concept_lower for word in ['pitch', 'era', 'whip', 'strikeout']):
                    pitching_score += 2
                if any(word in concept_lower for word in ['hit', 'batting', 'avg', 'hr', 'rbi']):
                    batting_score += 2
        
        # Make decision
        if pitching_score > batting_score:
            endpoint = "players/pitching"
            reasoning = f"Query contains {pitching_score} pitching-related terms vs {batting_score} batting terms"
        elif batting_score > pitching_score:
            endpoint = "players/batting"
            reasoning = f"Query contains {batting_score} batting-related terms vs {pitching_score} pitching terms"
        else:
            # Default to general players endpoint if unclear
            endpoint = "players"
            reasoning = "Query doesn't clearly favor pitching or batting stats"
        
        # Add reasoning to the intelligent query for transparency
        if not hasattr(iq, 'endpoint_reasoning'):
            iq.endpoint_reasoning = reasoning
            
        return endpoint
    
    def _calculate_confidence(self, iq: IntelligentQuery) -> float:
        """Calculate overall confidence in our understanding and approach"""
        
        confidences = []
        
        # Semantic confidence
        if iq.semantic_understanding:
            constraints = iq.semantic_understanding.get("constraints", [])
            concepts_found = iq.semantic_understanding.get("concepts_found", [])
            intent = iq.semantic_understanding.get("intent")
            
            semantic_conf = len(constraints) * 0.2  # More constraints = more confident
            semantic_conf += len(concepts_found) * 0.1  # More concepts = more confident
            semantic_conf += 0.5 if intent else 0.0
            confidences.append(min(1.0, semantic_conf))
        
        # Learning confidence
        if iq.learned_insights:
            confidences.append(iq.learned_insights.get("confidence", 0.0))
        
        # Contextual confidence (how well we understand the intent)
        if iq.contextual_intent:
            context_conf = 1.0 - iq.contextual_intent.urgency * 0.1  # Less confident when urgent
            context_conf += 0.2 if iq.contextual_intent.constraints else 0.0
            confidences.append(min(1.0, context_conf))
        
        # Overall confidence is the average
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def learn_from_feedback(self, query: str, data_found: Dict, user_rating: str):
        """Learn from user feedback to improve future performance"""
        self.knowledge_graph.observe_interaction(query, data_found, user_rating)
        
        # Save learned knowledge
        self.knowledge_graph.save_knowledge("learned_knowledge.json")
    
    def explain_reasoning(self, iq: IntelligentQuery) -> str:
        """Explain how the agent understood and plans to handle the query"""
        
        explanation = f"🧠 **How I understood your query:**\n\n"
        
        # Endpoint selection reasoning
        if hasattr(iq, 'endpoint_reasoning'):
            explanation += f"**Data Source Selection:**\n"
            explanation += f"- {iq.endpoint_reasoning}\n"
            explanation += f"- Selected endpoint: `{iq.execution_plan.get('endpoint', 'players')}`\n\n"
        
        # Semantic understanding
        if iq.semantic_understanding:
            sem = iq.semantic_understanding
            explanation += f"**Semantic Understanding:**\n"
            explanation += f"- Intent: {sem.get('intent', 'unknown')}\n"
            concepts_found = sem.get('concepts_found', [])
            if concepts_found:
                explanation += f"- Concepts found: {', '.join(concepts_found)}\n"
            constraints = sem.get('constraints', [])
            if constraints:
                explanation += f"- Constraints: {len(constraints)} filters identified\n"
            explanation += "\n"
        
        # Learned insights
        if iq.learned_insights and iq.learned_insights["confidence"] > 0.3:
            explanation += f"**Experience from Similar Queries:**\n"
            explanation += f"- Confidence: {iq.learned_insights['confidence']:.1%}\n"
            for reason in iq.learned_insights.get("reasoning", []):
                explanation += f"- {reason}\n"
            explanation += "\n"
        
        # Contextual understanding
        if iq.contextual_intent:
            context = iq.contextual_intent
            explanation += f"**Context Analysis:**\n"
            explanation += f"- Purpose: {context.purpose.value}\n"
            explanation += f"- Urgency: {context.urgency:.1f}/1.0\n"
            explanation += f"- Depth needed: {context.depth_required:.1f}/1.0\n"
            if context.implicit_knowledge:
                explanation += f"- Assumptions: {len(context.implicit_knowledge)} identified\n"
            explanation += "\n"
        
        # Execution plan
        explanation += f"**Execution Strategy:**\n"
        for step in iq.execution_plan.get("reasoning_chain", []):
            explanation += f"- {step}\n"
        
        explanation += f"\n**Overall Confidence: {iq.confidence:.1%}**"
        
        return explanation

# Integration functions for your main app.py
def replace_parse_query_intent(agent: IntelligentAgent, query: str, conversation_history: List[str] = None) -> Dict:
    """
    Drop-in replacement for your old parse_query_intent function.
    Returns the same structure but with intelligent processing.
    """
    
    intelligent_query = agent.process_query(query, conversation_history)
    
    # Get semantic understanding
    semantic = intelligent_query.semantic_understanding or {}
    concepts = semantic.get("concepts_found", [])
    target_concept = concepts[0] if concepts else "unknown"
    
    # Convert to format your existing app expects
    result = {
        "intent": "statistical_query",  # Your app expects this field
        "target_stat": target_concept,
        "filters": [],
        "sort_direction": "desc",
        "confidence": intelligent_query.confidence,
        "execution_plan": intelligent_query.execution_plan,
        "reasoning": intelligent_query.execution_plan.get("reasoning_chain", []),
    }
    
    # Extract filters from execution plan
    for filter_spec in intelligent_query.execution_plan.get("table_processing", {}).get("filters", []):
        if filter_spec["type"] == "minimum":
            result["filters"].append({
                "column": filter_spec["column_concept"],
                "operator": ">=",
                "value": filter_spec["value"]
            })
    
    # Extract sorting
    sorting = intelligent_query.execution_plan.get("table_processing", {}).get("sorting")
    if sorting:
        result["sort_direction"] = sorting["direction"]
        result["sort_column"] = sorting["concept"]
    
    return result

# Example usage
if __name__ == "__main__":
    # Create the intelligent agent
    agent = IntelligentAgent()
    
    # Test queries
    test_queries = [
        "which player has the highest CVR with at least 80 games",
        "who leads in batting average among qualified players",
        "I need the best home run hitters for my fantasy lineup"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Process with full intelligence
        intelligent_query = agent.process_query(query)
        
        # Show the explanation
        explanation = agent.explain_reasoning(intelligent_query)
        print(explanation)
        
        # Show the execution plan
        print(f"\n**Execution Plan:**")
        print(json.dumps(intelligent_query.execution_plan, indent=2))
