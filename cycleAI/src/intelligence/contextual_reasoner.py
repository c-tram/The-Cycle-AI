"""
Contextual Reasoning Engine
Understands WHY users ask questions, not just WHAT they ask.
"""

from typing import Dict, List, Optional, Tuple
import re
from dataclasses import dataclass
from enum import Enum

class QueryPurpose(Enum):
    """Why is the user asking this question?"""
    SCOUTING = "scouting"  # Evaluating player talent
    COMPARISON = "comparison"  # Comparing players
    FANTASY = "fantasy"  # Fantasy baseball decisions
    RESEARCH = "research"  # General baseball research
    VALIDATION = "validation"  # Checking a hypothesis
    DISCOVERY = "discovery"  # Looking for patterns/insights

class Context(Enum):
    """What context is the user operating in?"""
    CURRENT_SEASON = "current_season"
    HISTORICAL = "historical"
    CAREER_ANALYSIS = "career"
    SITUATIONAL = "situational"  # Specific game situations
    DEVELOPMENTAL = "development"  # Player development

@dataclass
class UserIntent:
    """Rich understanding of what the user actually wants"""
    purpose: QueryPurpose
    context: Context
    urgency: float  # 0.0 = casual, 1.0 = urgent decision needed
    depth_required: float  # 0.0 = surface level, 1.0 = deep analysis
    confidence_threshold: float  # How certain does answer need to be?
    constraints: Dict[str, any]  # Filters, minimums, etc.
    implicit_knowledge: List[str]  # What the user assumes we know

class ContextualReasoner:
    """Reasons about user intent and context"""
    
    def __init__(self):
        # Baseball domain knowledge
        self.stat_purposes = {
            "cvr": {"primary": QueryPurpose.SCOUTING, "context": Context.CURRENT_SEASON},
            "avg": {"primary": QueryPurpose.COMPARISON, "context": Context.CURRENT_SEASON},
            "era": {"primary": QueryPurpose.SCOUTING, "context": Context.CURRENT_SEASON},
            "war": {"primary": QueryPurpose.RESEARCH, "context": Context.CAREER_ANALYSIS},
            "ops": {"primary": QueryPurpose.COMPARISON, "context": Context.CURRENT_SEASON},
        }
        
        # Context clues from language patterns
        self.urgency_indicators = {
            "need to know": 0.8,
            "urgent": 0.9,
            "quickly": 0.7,
            "asap": 0.9,
            "just curious": 0.2,
            "wondering": 0.3,
            "interested in": 0.4
        }
        
        self.depth_indicators = {
            "deep dive": 0.9,
            "analysis": 0.7,
            "detailed": 0.8,
            "comprehensive": 0.8,
            "quick": 0.2,
            "simple": 0.2,
            "just": 0.1,  # "just want to know"
            "overview": 0.3
        }
        
    def understand_intent(self, query: str, conversation_history: List[str] = None) -> UserIntent:
        """Deeply understand what the user actually wants"""
        query_lower = query.lower()
        
        # Determine primary purpose
        purpose = self._identify_purpose(query_lower)
        
        # Determine context
        context = self._identify_context(query_lower)
        
        # Assess urgency from language patterns
        urgency = self._assess_urgency(query_lower)
        
        # Determine depth required
        depth = self._assess_depth_required(query_lower)
        
        # Set confidence threshold based on purpose
        confidence_threshold = self._determine_confidence_threshold(purpose, query_lower)
        
        # Extract constraints
        constraints = self._extract_constraints(query_lower)
        
        # Identify implicit knowledge
        implicit_knowledge = self._identify_implicit_knowledge(query_lower, conversation_history)
        
        return UserIntent(
            purpose=purpose,
            context=context,
            urgency=urgency,
            depth_required=depth,
            confidence_threshold=confidence_threshold,
            constraints=constraints,
            implicit_knowledge=implicit_knowledge
        )
    
    def _identify_purpose(self, query: str) -> QueryPurpose:
        """Why is the user asking this?"""
        
        # Scouting indicators
        if any(word in query for word in ["best", "top", "highest", "talent", "prospect"]):
            return QueryPurpose.SCOUTING
        
        # Comparison indicators
        if any(word in query for word in ["vs", "versus", "compare", "better than", "difference"]):
            return QueryPurpose.COMPARISON
        
        # Fantasy indicators
        if any(word in query for word in ["start", "sit", "lineup", "draft", "pickup"]):
            return QueryPurpose.FANTASY
        
        # Validation indicators
        if any(word in query for word in ["is", "does", "confirm", "true that", "correct"]):
            return QueryPurpose.VALIDATION
        
        # Discovery indicators  
        if any(word in query for word in ["trend", "pattern", "correlation", "insight", "find"]):
            return QueryPurpose.DISCOVERY
        
        return QueryPurpose.RESEARCH  # Default
    
    def _identify_context(self, query: str) -> Context:
        """What timeframe/context is relevant?"""
        
        if any(word in query for word in ["this season", "2024", "current", "now", "today"]):
            return Context.CURRENT_SEASON
        
        if any(word in query for word in ["career", "lifetime", "all time", "ever"]):
            return Context.CAREER_ANALYSIS
        
        if any(word in query for word in ["historical", "past", "was", "used to", "history"]):
            return Context.HISTORICAL
        
        if any(word in query for word in ["clutch", "bases loaded", "late inning", "pressure"]):
            return Context.SITUATIONAL
        
        if any(word in query for word in ["developing", "improving", "progress", "growth"]):
            return Context.DEVELOPMENTAL
        
        return Context.CURRENT_SEASON  # Default assumption
    
    def _assess_urgency(self, query: str) -> float:
        """How urgently does the user need this answer?"""
        urgency = 0.5  # Baseline
        
        for indicator, weight in self.urgency_indicators.items():
            if indicator in query:
                urgency = max(urgency, weight)
        
        # Questions marks often indicate casual inquiry
        if query.count('?') > 1:
            urgency *= 0.8
        
        return urgency
    
    def _assess_depth_required(self, query: str) -> float:
        """How detailed should the answer be?"""
        depth = 0.5  # Baseline
        
        for indicator, weight in self.depth_indicators.items():
            if indicator in query:
                depth = max(depth, weight)
        
        # Multiple stats requested = deeper analysis needed
        stat_count = len([s for s in ["cvr", "avg", "era", "war", "ops"] if s in query])
        depth += min(0.3, stat_count * 0.1)
        
        return min(1.0, depth)
    
    def _determine_confidence_threshold(self, purpose: QueryPurpose, query: str) -> float:
        """How confident do we need to be in our answer?"""
        
        # High-stakes purposes need high confidence
        if purpose == QueryPurpose.FANTASY:
            return 0.8  # Fantasy decisions have consequences
        elif purpose == QueryPurpose.SCOUTING:
            return 0.75  # Scouting affects real decisions
        elif purpose == QueryPurpose.VALIDATION:
            return 0.9  # Validation needs to be very sure
        else:
            return 0.6  # Research/comparison can be more speculative
    
    def _extract_constraints(self, query: str) -> Dict[str, any]:
        """Extract filters, minimums, requirements"""
        constraints = {}
        
        # Games played constraints
        games_match = re.search(r'(at least|minimum|more than|\+)\s*(\d+)\s*(games?|g\b)', query)
        if games_match:
            constraints['min_games'] = int(games_match.group(2))
        
        # Statistical thresholds
        stat_match = re.search(r'(\w+)\s*(over|above|more than)\s*([0-9.]+)', query)
        if stat_match:
            constraints[f'min_{stat_match.group(1)}'] = float(stat_match.group(3))
        
        # Team constraints
        team_match = re.search(r'(on|for|with)\s+(the\s+)?([A-Z][a-z]+\s*[A-Z]*[a-z]*)', query)
        if team_match and len(team_match.group(3)) > 2:  # Not just abbreviation
            constraints['team'] = team_match.group(3)
        
        # Position constraints
        positions = ['pitcher', 'catcher', 'infield', 'outfield', 'dh']
        for pos in positions:
            if pos in query:
                constraints['position'] = pos
        
        return constraints
    
    def _identify_implicit_knowledge(self, query: str, history: List[str] = None) -> List[str]:
        """What does the user assume we already know?"""
        implicit = []
        
        # Baseball knowledge assumptions
        if "qualified" in query:
            implicit.append("User assumes we know plate appearance minimums for qualification")
        
        if "cvr" in query and "contact value" not in query:
            implicit.append("User assumes we know CVR = Contact Value Rating")
        
        if any(team in query for team in ["yankees", "dodgers", "astros"]):
            implicit.append("User assumes we know team contexts and rivalries")
        
        # Statistical literacy assumptions
        if any(advanced in query for advanced in ["war", "wrc+", "fip", "xwoba"]):
            implicit.append("User assumes we understand advanced metrics")
        
        # Context from conversation history
        if history:
            for prev_query in history[-3:]:  # Last 3 queries
                if "season" in prev_query:
                    implicit.append("Context: Previous queries about current season")
                if any(stat in prev_query for stat in ["cvr", "avg", "era"]):
                    implicit.append("Context: User interested in statistical analysis")
        
        return implicit
    
    def generate_reasoning_strategy(self, intent: UserIntent) -> Dict[str, any]:
        """Generate a strategy for answering based on understood intent"""
        strategy = {
            "data_freshness_required": 0.5,
            "explanation_depth": intent.depth_required,
            "confidence_communication": intent.confidence_threshold > 0.7,
            "context_setting": [],
            "followup_suggestions": [],
            "answer_style": "informative"
        }
        
        # Adjust based on purpose
        if intent.purpose == QueryPurpose.FANTASY:
            strategy["answer_style"] = "actionable"
            strategy["data_freshness_required"] = 0.9
            strategy["followup_suggestions"].append("Would you like injury status for these players?")
            
        elif intent.purpose == QueryPurpose.SCOUTING:
            strategy["answer_style"] = "analytical"
            strategy["context_setting"].append("Current season performance context")
            strategy["followup_suggestions"].append("Want to see these players' recent game logs?")
            
        elif intent.purpose == QueryPurpose.COMPARISON:
            strategy["answer_style"] = "comparative"
            strategy["context_setting"].append("Head-to-head statistical comparison")
            
        # Adjust based on urgency
        if intent.urgency > 0.7:
            strategy["answer_style"] = "direct"
            strategy["explanation_depth"] *= 0.7  # Less verbose when urgent
        
        # Add confidence communication
        if intent.confidence_threshold > 0.8:
            strategy["confidence_communication"] = True
            strategy["context_setting"].append("Data reliability and sample size notes")
        
        return strategy

# Testing the contextual reasoning
if __name__ == "__main__":
    reasoner = ContextualReasoner()
    
    test_queries = [
        "I need to know who has the highest CVR with at least 80 games - have a fantasy decision to make ASAP",
        "Just curious about batting average leaders this season",
        "Can you do a deep analysis comparing Judge vs. Ohtani's war this year?",
        "Is it true that contact value rating correlates with clutch performance?",
        "Who are the top prospects showing development in power hitting?"
    ]
    
    for query in test_queries:
        intent = reasoner.understand_intent(query)
        strategy = reasoner.generate_reasoning_strategy(intent)
        
        print(f"\nQuery: {query}")
        print(f"Purpose: {intent.purpose.value}")
        print(f"Context: {intent.context.value}")
        print(f"Urgency: {intent.urgency:.2f}")
        print(f"Depth: {intent.depth_required:.2f}")
        print(f"Confidence needed: {intent.confidence_threshold:.2f}")
        print(f"Answer style: {strategy['answer_style']}")
        if intent.constraints:
            print(f"Constraints: {intent.constraints}")
        if intent.implicit_knowledge:
            print("Implicit assumptions:")
            for assumption in intent.implicit_knowledge:
                print(f"  - {assumption}")
