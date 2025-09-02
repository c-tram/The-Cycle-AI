"""
Self-Learning Knowledge System
Builds understanding through observation and feedback, not just rules.
"""

import json
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import re

class KnowledgeNode:
    """Represents a piece of learned knowledge"""
    def __init__(self, concept: str, context: str = ""):
        self.concept = concept
        self.context = context
        self.connections = defaultdict(float)  # concept -> strength
        self.evidence = []  # Examples that support this knowledge
        self.confidence = 0.0
        self.usage_count = 0
    
    def add_evidence(self, evidence: str, outcome: str):
        """Add evidence that supports or refutes this knowledge"""
        self.evidence.append({"evidence": evidence, "outcome": outcome})
        self.update_confidence()
    
    def update_confidence(self):
        """Calculate confidence based on evidence"""
        if not self.evidence:
            self.confidence = 0.0
            return
        
        positive_evidence = sum(1 for e in self.evidence if e["outcome"] == "success")
        total_evidence = len(self.evidence)
        
        # Confidence grows with positive evidence and usage
        base_confidence = positive_evidence / total_evidence
        usage_bonus = min(0.2, self.usage_count / 100)  # Cap at 20% bonus
        self.confidence = min(1.0, base_confidence + usage_bonus)

class LearningKnowledgeGraph:
    """A knowledge graph that learns from interactions"""
    
    def __init__(self):
        self.nodes = {}  # concept -> KnowledgeNode
        self.interaction_history = []
        self.concept_patterns = defaultdict(list)  # pattern -> [successful concepts]
        self.column_mappings = defaultdict(set)  # canonical_name -> {aliases}
        
    def observe_interaction(self, query: str, data_found: Dict, user_rating: str):
        """Learn from a user interaction"""
        self.interaction_history.append({
            "query": query,
            "data": data_found,
            "rating": user_rating,
            "timestamp": len(self.interaction_history)
        })
        
        # Extract learning from this interaction
        self.extract_column_knowledge(data_found, user_rating)
        self.extract_query_patterns(query, data_found, user_rating)
        self.update_concept_relationships(query, data_found, user_rating)
    
    def extract_column_knowledge(self, data_found: Dict, rating: str):
        """Learn about column name variations"""
        if "tables" not in data_found:
            return
            
        for table in data_found["tables"]:
            columns = table.get("columns", [])
            
            # Look for game-related columns
            game_columns = [col for col in columns if any(g in col.lower() for g in ["g", "game", "gp"])]
            if game_columns:
                for col in game_columns:
                    self.column_mappings["games_played"].add(col)
                    
                    # Create or update knowledge node
                    node_key = f"column_alias_{col}"
                    if node_key not in self.nodes:
                        self.nodes[node_key] = KnowledgeNode(f"column_alias", f"{col} represents games played")
                    
                    self.nodes[node_key].add_evidence(f"Column '{col}' found in table", rating)
            
            # Look for stat columns
            stat_indicators = ["cvr", "avg", "era", "hr", "rbi"]
            for indicator in stat_indicators:
                matching_cols = [col for col in columns if indicator in col.lower()]
                for col in matching_cols:
                    self.column_mappings[indicator].add(col)
    
    def extract_query_patterns(self, query: str, data_found: Dict, rating: str):
        """Learn successful query patterns"""
        query_lower = query.lower()
        
        # Extract key phrases
        key_phrases = []
        
        # Statistical queries
        if any(word in query_lower for word in ["highest", "most", "best", "top"]):
            key_phrases.append("ranking_request")
        
        if "at least" in query_lower:
            key_phrases.append("minimum_threshold")
            
        if any(stat in query_lower for stat in ["cvr", "batting", "era", "home run"]):
            key_phrases.append("statistical_query")
        
        # Record which patterns lead to success
        for phrase in key_phrases:
            self.concept_patterns[phrase].append({
                "query": query,
                "success": rating == "good",
                "data_found": len(data_found.get("tables", []))
            })
    
    def update_concept_relationships(self, query: str, data_found: Dict, rating: str):
        """Learn relationships between concepts"""
        # Extract concepts from query
        concepts = self.extract_concepts_from_query(query)
        
        # If successful, strengthen relationships between co-occurring concepts
        if rating == "good":
            for i, concept1 in enumerate(concepts):
                for concept2 in concepts[i+1:]:
                    # Strengthen bidirectional relationship
                    if concept1 not in self.nodes:
                        self.nodes[concept1] = KnowledgeNode(concept1)
                    if concept2 not in self.nodes:
                        self.nodes[concept2] = KnowledgeNode(concept2)
                    
                    self.nodes[concept1].connections[concept2] += 0.1
                    self.nodes[concept2].connections[concept1] += 0.1
    
    def extract_concepts_from_query(self, query: str) -> List[str]:
        """Extract baseball concepts from a query"""
        concepts = []
        query_lower = query.lower()
        
        # Statistical concepts
        stats = ["cvr", "batting average", "era", "home runs", "rbi", "games"]
        for stat in stats:
            if stat in query_lower:
                concepts.append(stat.replace(" ", "_"))
        
        # Action concepts  
        actions = ["highest", "lowest", "most", "least", "best", "worst"]
        for action in actions:
            if action in query_lower:
                concepts.append(f"find_{action}")
        
        # Constraint concepts
        if "at least" in query_lower:
            concepts.append("minimum_constraint")
        
        return concepts
    
    def predict_best_approach(self, query: str) -> Dict:
        """Use learned knowledge to predict best approach"""
        query_lower = query.lower()
        prediction = {
            "confidence": 0.0,
            "suggested_columns": [],
            "suggested_filters": [],
            "data_source": "https://thecycle.online/players",
            "reasoning": []
        }
        
        # Use learned column mappings
        if any(word in query_lower for word in ["game", "games"]):
            best_game_column = self.get_most_reliable_column_alias("games_played")
            if best_game_column:
                prediction["suggested_columns"].append(best_game_column)
                prediction["reasoning"].append(f"Learned that '{best_game_column}' is reliable for games data")
        
        # Use pattern learning
        relevant_patterns = []
        for pattern, examples in self.concept_patterns.items():
            if self.pattern_matches_query(pattern, query_lower):
                success_rate = sum(1 for ex in examples if ex["success"]) / len(examples) if examples else 0
                relevant_patterns.append((pattern, success_rate))
        
        if relevant_patterns:
            best_pattern, confidence = max(relevant_patterns, key=lambda x: x[1])
            prediction["confidence"] = confidence
            prediction["reasoning"].append(f"Pattern '{best_pattern}' has {confidence:.1%} success rate")
        
        # Use concept relationships
        query_concepts = self.extract_concepts_from_query(query)
        for concept in query_concepts:
            if concept in self.nodes:
                node = self.nodes[concept]
                prediction["confidence"] = max(prediction["confidence"], node.confidence)
                
                # Suggest related concepts
                for related_concept, strength in node.connections.items():
                    if strength > 0.5:  # Strong relationship
                        prediction["reasoning"].append(f"'{concept}' strongly related to '{related_concept}'")
        
        return prediction
    
    def get_most_reliable_column_alias(self, canonical_name: str) -> str:
        """Get the most reliable column name for a concept"""
        aliases = self.column_mappings.get(canonical_name, set())
        if not aliases:
            return None
        
        # Score aliases by reliability
        alias_scores = {}
        for alias in aliases:
            node_key = f"column_alias_{alias}"
            if node_key in self.nodes:
                alias_scores[alias] = self.nodes[node_key].confidence
            else:
                alias_scores[alias] = 0.0
        
        return max(alias_scores, key=alias_scores.get) if alias_scores else None
    
    def pattern_matches_query(self, pattern: str, query: str) -> bool:
        """Check if a learned pattern matches the current query"""
        pattern_keywords = {
            "ranking_request": ["highest", "most", "best", "top", "leader"],
            "minimum_threshold": ["at least", "minimum", "more than"],
            "statistical_query": ["cvr", "average", "era", "home run", "stat"]
        }
        
        keywords = pattern_keywords.get(pattern, [])
        return any(keyword in query for keyword in keywords)
    
    def save_knowledge(self, filepath: str):
        """Save learned knowledge to file"""
        knowledge_data = {
            "column_mappings": {k: list(v) for k, v in self.column_mappings.items()},
            "concept_patterns": dict(self.concept_patterns),
            "interaction_count": len(self.interaction_history),
            "nodes": {
                node_id: {
                    "concept": node.concept,
                    "context": node.context,
                    "connections": dict(node.connections),
                    "confidence": node.confidence,
                    "usage_count": node.usage_count
                } for node_id, node in self.nodes.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(knowledge_data, f, indent=2)
    
    def load_knowledge(self, filepath: str):
        """Load previously learned knowledge"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore column mappings
            self.column_mappings = defaultdict(set)
            for canonical, aliases in data.get("column_mappings", {}).items():
                self.column_mappings[canonical] = set(aliases)
            
            # Restore patterns
            self.concept_patterns = defaultdict(list, data.get("concept_patterns", {}))
            
            # Restore nodes
            for node_id, node_data in data.get("nodes", {}).items():
                node = KnowledgeNode(node_data["concept"], node_data["context"])
                node.connections = defaultdict(float, node_data["connections"])
                node.confidence = node_data["confidence"]
                node.usage_count = node_data["usage_count"]
                self.nodes[node_id] = node
                
        except FileNotFoundError:
            pass  # Start with empty knowledge base

# Integration example
if __name__ == "__main__":
    # Create learning system
    kg = LearningKnowledgeGraph()
    
    # Simulate learning from interactions
    interactions = [
        {
            "query": "which player has the highest CVR with at least 80 games",
            "data": {"tables": [{"columns": ["Name", "Team", "G", "CVR", "AB"]}]},
            "rating": "good"
        },
        {
            "query": "who leads in batting average among qualified players", 
            "data": {"tables": [{"columns": ["Player", "Team", "Games", "AVG", "OBP"]}]},
            "rating": "good"
        },
        {
            "query": "best home run hitters with 100+ games",
            "data": {"tables": [{"columns": ["Name", "Team", "GP", "HR", "RBI"]}]},
            "rating": "bad"  # Maybe didn't filter properly
        }
    ]
    
    # Learn from interactions
    for interaction in interactions:
        kg.observe_interaction(
            interaction["query"], 
            interaction["data"], 
            interaction["rating"]
        )
    
    # Test prediction
    test_query = "highest CVR value with at least 90 games played"
    prediction = kg.predict_best_approach(test_query)
    
    print(f"Query: {test_query}")
    print(f"Prediction confidence: {prediction['confidence']:.2f}")
    print(f"Suggested columns: {prediction['suggested_columns']}")
    print("Reasoning:")
    for reason in prediction["reasoning"]:
        print(f"  - {reason}")
    
    # Save learned knowledge
    kg.save_knowledge("learned_knowledge.json")
