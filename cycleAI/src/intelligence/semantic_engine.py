"""
Semantic Understanding Engine - Your Own Intelligence System
This builds conceptual understanding rather than just pattern matching.
"""

import json
import re
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Concept:
    """Represents a baseball concept with relationships"""
    name: str
    aliases: Set[str]
    category: str  # 'stat', 'position', 'team', 'action', etc.
    relationships: Dict[str, List[str]]  # 'synonyms', 'related_to', 'measured_by'
    data_locations: List[str]  # Where to find this data
    
class SemanticEngine:
    """Builds understanding of baseball concepts and their relationships"""
    
    def __init__(self):
        self.concepts = {}
        self.concept_graph = defaultdict(set)
        self.load_baseball_knowledge()
    
    def load_baseball_knowledge(self):
        """Build fundamental baseball knowledge graph"""
        
        # Core statistical concepts
        self.add_concept(Concept(
            name="games_played",
            aliases={"G", "games", "GP", "games_played", "game count"},
            category="stat",
            relationships={
                "measures": ["playing_time", "availability"],
                "affects": ["per_game_stats", "reliability"],
                "threshold_for": ["qualified_player"]
            },
            data_locations=["table_columns", "player_cards"]
        ))
        
        self.add_concept(Concept(
            name="contact_value_rating",
            aliases={"CVR", "contact rating", "contact value", "hitting skill"},
            category="stat",
            relationships={
                "measures": ["batting_ability", "contact_quality"],
                "higher_is": ["better"],
                "calculated_from": ["contact_rate", "swing_decisions"]
            },
            data_locations=["advanced_stats_table"]
        ))
        
        # Query intent concepts
        self.add_concept(Concept(
            name="leadership_query",
            aliases={"who leads", "highest", "best", "most", "top", "leader"},
            category="intent",
            relationships={
                "requires": ["sorting", "ranking"],
                "implies": ["comparative_analysis"]
            },
            data_locations=[]
        ))
        
        # Build concept relationships
        self.build_concept_graph()
    
    def add_concept(self, concept: Concept):
        """Add a concept to the knowledge base"""
        self.concepts[concept.name] = concept
        
        # Index all aliases
        for alias in concept.aliases:
            self.concept_graph[alias.lower()].add(concept.name)
    
    def build_concept_graph(self):
        """Build relationships between concepts"""
        for concept_name, concept in self.concepts.items():
            for rel_type, related_concepts in concept.relationships.items():
                for related in related_concepts:
                    self.concept_graph[f"{concept_name}_{rel_type}"].add(related)
    
    def understand_query(self, query: str) -> Dict:
        """Extract semantic meaning from natural language"""
        query_lower = query.lower()
        understanding = {
            "concepts_found": [],
            "intent": None,
            "constraints": [],
            "semantic_meaning": None
        }
        
        # Find concepts mentioned in query
        for term in re.findall(r'\b\w+\b', query_lower):
            if term in self.concept_graph:
                concept_names = list(self.concept_graph[term])
                understanding["concepts_found"].extend(concept_names)
        
        # Extract constraints (like "at least 80 games")
        constraints = self.extract_constraints(query_lower)
        understanding["constraints"] = constraints
        
        # Determine intent
        intent = self.determine_intent(query_lower, understanding["concepts_found"])
        understanding["intent"] = intent
        
        # Build semantic meaning
        understanding["semantic_meaning"] = self.build_semantic_meaning(understanding)
        
        return understanding
    
    def extract_constraints(self, query: str) -> List[Dict]:
        """Extract logical constraints from query"""
        constraints = []
        
        # Numerical constraints
        at_least_pattern = r'at least (\d+) (\w+)'
        matches = re.findall(at_least_pattern, query)
        for value, unit in matches:
            constraints.append({
                "type": "minimum_threshold",
                "value": int(value),
                "applies_to": self.resolve_concept(unit),
                "operator": ">="
            })
        
        return constraints
    
    def determine_intent(self, query: str, concepts: List[str]) -> Dict:
        """Determine what the user wants to accomplish"""
        if any(concept in ["leadership_query"] for concept in concepts):
            return {
                "type": "ranking_query",
                "action": "find_leader",
                "requires_sorting": True
            }
        
        if "compare" in query or "vs" in query:
            return {
                "type": "comparison_query", 
                "action": "compare_entities"
            }
        
        return {"type": "information_query", "action": "retrieve_data"}
    
    def resolve_concept(self, term: str) -> str:
        """Resolve aliases to canonical concept names"""
        term_lower = term.lower()
        if term_lower in self.concept_graph:
            concept_names = list(self.concept_graph[term_lower])
            return concept_names[0] if concept_names else term
        return term
    
    def build_semantic_meaning(self, understanding: Dict) -> Dict:
        """Build high-level semantic understanding"""
        concepts = understanding["concepts_found"]
        intent = understanding["intent"]
        constraints = understanding["constraints"]
        
        if intent and intent["type"] == "ranking_query":
            # This is a "find the best player" type query
            stat_concepts = [c for c in concepts if c in self.concepts and self.concepts[c].category == "stat"]
            
            if stat_concepts and constraints:
                return {
                    "query_type": "constrained_ranking",
                    "rank_by": stat_concepts[0],  # Primary stat to rank by
                    "filter_by": constraints[0]["applies_to"] if constraints else None,
                    "filter_condition": constraints[0] if constraints else None,
                    "data_strategy": "filter_then_sort"
                }
        
        return {"query_type": "general", "data_strategy": "retrieve_all"}
    
    def get_data_strategy(self, semantic_meaning: Dict) -> Dict:
        """Convert semantic understanding into data retrieval strategy"""
        if semantic_meaning["query_type"] == "constrained_ranking":
            return {
                "source": "https://thecycle.online/players",
                "processing_steps": [
                    {
                        "step": "parse_tables",
                        "action": "extract_all_tables"
                    },
                    {
                        "step": "apply_filters",
                        "filters": [semantic_meaning.get("filter_condition")]
                    },
                    {
                        "step": "sort_data",
                        "sort_by": semantic_meaning.get("rank_by"),
                        "order": "descending"
                    },
                    {
                        "step": "return_top_results",
                        "limit": 5
                    }
                ]
            }
        
        return {"source": "general_search", "processing_steps": []}

# Example usage and testing
if __name__ == "__main__":
    engine = SemanticEngine()
    
    # Test queries
    test_queries = [
        "which player has the highest CVR value with at least 80 games played",
        "who leads in home runs",
        "best batting average among qualified players"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        understanding = engine.understand_query(query)
        strategy = engine.get_data_strategy(understanding["semantic_meaning"])
        
        print(f"Concepts: {understanding['concepts_found']}")
        print(f"Intent: {understanding['intent']}")
        print(f"Constraints: {understanding['constraints']}")
        print(f"Semantic Meaning: {understanding['semantic_meaning']}")
        print(f"Data Strategy: {strategy}")
