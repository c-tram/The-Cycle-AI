"""
Autonomous Prompt Library Generator for Baseball AI
This system automatically generates, tests, and refines baseball queries
"""

import json
import random
import itertools
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os
import sys
from dataclasses import dataclass, asdict
import requests

# Add shared utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.baseball_utils import BaseballDataManager
from ..advanced.langchain_baseball_agent import LangChainBaseballAgent

@dataclass
class PromptTemplate:
    """Template for generating baseball prompts"""
    category: str
    pattern: str
    variables: Dict[str, List[str]]
    complexity: str  # 'simple', 'medium', 'complex'
    expected_tools: List[str]
    description: str

@dataclass
class GeneratedPrompt:
    """A generated prompt with metadata"""
    prompt: str
    template_id: str
    category: str
    complexity: str
    variables_used: Dict[str, str]
    timestamp: str
    tested: bool = False
    response_quality: float = 0.0
    tools_used: List[str] = None
    response_length: int = 0
    success: bool = False

class AutonomousPromptGenerator:
    """Generates and manages baseball prompts autonomously"""
    
    def __init__(self):
        self.data_manager = BaseballDataManager()
        self.agent = LangChainBaseballAgent()
        self.prompt_templates = self._initialize_templates()
        self.generated_prompts: List[GeneratedPrompt] = []
        self.prompt_library = {}
        
    def _initialize_templates(self) -> List[PromptTemplate]:
        """Initialize comprehensive prompt templates"""
        return [
            # Player Statistics Templates
            PromptTemplate(
                category="player_stats",
                pattern="Who has the most {stat} in {timeframe}?",
                variables={
                    "stat": ["home runs", "RBIs", "hits", "batting average", "stolen bases", "strikeouts"],
                    "timeframe": ["this season", "2024", "their career", "the last month"]
                },
                complexity="simple",
                expected_tools=["fetch_players_data"],
                description="Basic player statistical queries"
            ),
            
            PromptTemplate(
                category="player_comparison",
                pattern="Compare {player1} and {player2} in terms of {stat_category}",
                variables={
                    "player1": ["Aaron Judge", "Mookie Betts", "Ronald Acuna Jr.", "Mike Trout"],
                    "player2": ["Freddie Freeman", "Jose Altuve", "Vladimir Guerrero Jr.", "Juan Soto"],
                    "stat_category": ["batting stats", "power numbers", "overall performance", "clutch hitting"]
                },
                complexity="medium",
                expected_tools=["fetch_players_data"],
                description="Player comparison queries"
            ),
            
            PromptTemplate(
                category="team_analysis",
                pattern="How is the {team} performing in {aspect} this season?",
                variables={
                    "team": ["Yankees", "Dodgers", "Astros", "Braves", "Mets", "Red Sox"],
                    "aspect": ["batting", "pitching", "defense", "overall record", "recent games"]
                },
                complexity="medium",
                expected_tools=["fetch_teams_data", "analyze_team_performance"],
                description="Team performance analysis"
            ),
            
            PromptTemplate(
                category="filtered_searches",
                pattern="Show me all players whose last name starts with {letter} who have more than {number} {stat}",
                variables={
                    "letter": ["A", "B", "C", "D", "T", "S", "M"],
                    "number": ["10", "20", "30", "50", "100"],
                    "stat": ["home runs", "RBIs", "hits", "stolen bases"]
                },
                complexity="complex",
                expected_tools=["filter_players_by_last_name", "fetch_players_data"],
                description="Complex filtered player searches"
            ),
            
            PromptTemplate(
                category="best_team",
                pattern="Build me the best {type} team with {constraint}",
                variables={
                    "type": ["offensive", "defensive", "balanced", "power hitting", "contact hitting"],
                    "constraint": ["every position filled", "players under 30", "veteran players", "rookies only"]
                },
                complexity="complex",
                expected_tools=["build_best_team", "fetch_players_data"],
                description="Team building with constraints"
            ),
            
            PromptTemplate(
                category="league_analysis",
                pattern="Which {league} team has the best {category} and why?",
                variables={
                    "league": ["American League", "National League", "AL East", "NL West"],
                    "category": ["batting average", "home run production", "pitching staff", "bullpen", "defense"]
                },
                complexity="medium",
                expected_tools=["fetch_teams_data", "analyze_team_performance"],
                description="League-wide analysis"
            ),
            
            PromptTemplate(
                category="historical_context",
                pattern="How does {player}'s {stat} compare to historical {benchmark}?",
                variables={
                    "player": ["Aaron Judge", "Shohei Ohtani", "Mike Trout", "Ronald Acuna Jr."],
                    "stat": ["home run total", "batting average", "RBI production", "stolen base count"],
                    "benchmark": ["legends", "modern era players", "Hall of Famers", "MVP winners"]
                },
                complexity="complex",
                expected_tools=["fetch_players_data"],
                description="Historical context comparisons"
            ),
            
            PromptTemplate(
                category="situational_analysis",
                pattern="Who performs best in {situation} for the {team}?",
                variables={
                    "situation": ["clutch situations", "with runners in scoring position", "late innings", "high leverage"],
                    "team": ["Yankees", "Dodgers", "Astros", "Braves", "Red Sox", "Mets"]
                },
                complexity="complex",
                expected_tools=["fetch_players_data", "analyze_team_performance"],
                description="Situational performance analysis"
            ),
            
            PromptTemplate(
                category="prospect_analysis",
                pattern="Tell me about the {position} prospects in the {team} system",
                variables={
                    "position": ["pitching", "hitting", "catcher", "shortstop", "outfield"],
                    "team": ["Yankees", "Dodgers", "Braves", "Padres", "Orioles", "Tigers"]
                },
                complexity="medium",
                expected_tools=["fetch_players_data"],
                description="Prospect and development analysis"
            ),
            
            PromptTemplate(
                category="advanced_metrics",
                pattern="Rank the top 10 {position} by {advanced_stat} and explain the results",
                variables={
                    "position": ["starting pitchers", "relief pitchers", "catchers", "first basemen", "outfielders"],
                    "advanced_stat": ["OPS", "ERA+", "WHIP", "slugging percentage", "on-base percentage"]
                },
                complexity="complex",
                expected_tools=["fetch_players_data"],
                description="Advanced statistical analysis"
            )
        ]
    
    def generate_prompt_variations(self, num_prompts: int = 50) -> List[GeneratedPrompt]:
        """Generate diverse prompt variations"""
        generated = []
        
        for i in range(num_prompts):
            template = random.choice(self.prompt_templates)
            
            # Generate all combinations for this template
            variable_combinations = []
            var_names = list(template.variables.keys())
            var_values = [template.variables[name] for name in var_names]
            
            # Get a random combination
            combination = [random.choice(values) for values in var_values]
            variables_used = dict(zip(var_names, combination))
            
            # Format the prompt
            try:
                prompt = template.pattern.format(**variables_used)
                
                generated_prompt = GeneratedPrompt(
                    prompt=prompt,
                    template_id=f"{template.category}_{i}",
                    category=template.category,
                    complexity=template.complexity,
                    variables_used=variables_used,
                    timestamp=datetime.now().isoformat(),
                    tools_used=template.expected_tools.copy()
                )
                
                generated.append(generated_prompt)
                
            except KeyError as e:
                print(f"Template formatting error: {e}")
                continue
        
        self.generated_prompts.extend(generated)
        return generated
    
    def test_prompt_batch(self, prompts: List[GeneratedPrompt], max_tests: int = 10) -> Dict[str, Any]:
        """Test a batch of prompts and evaluate responses"""
        results = {
            'total_tested': 0,
            'successful': 0,
            'failed': 0,
            'avg_response_length': 0,
            'tool_usage': {},
            'category_performance': {},
            'complexity_performance': {}
        }
        
        tested_prompts = prompts[:max_tests]
        total_length = 0
        
        for prompt_obj in tested_prompts:
            try:
                print(f"\n🧪 Testing: {prompt_obj.prompt[:50]}...")
                
                # Test the prompt
                response = self.agent.process_query(prompt_obj.prompt)
                
                # Evaluate response quality
                quality_score = self._evaluate_response_quality(response, prompt_obj)
                response_length = len(response)
                
                # Update prompt object
                prompt_obj.tested = True
                prompt_obj.response_quality = quality_score
                prompt_obj.response_length = response_length
                prompt_obj.success = quality_score > 0.5
                
                # Update results
                results['total_tested'] += 1
                total_length += response_length
                
                if prompt_obj.success:
                    results['successful'] += 1
                    print(f"✅ Success (Quality: {quality_score:.2f})")
                else:
                    results['failed'] += 1
                    print(f"❌ Failed (Quality: {quality_score:.2f})")
                
                # Track tool usage
                for tool in prompt_obj.tools_used:
                    results['tool_usage'][tool] = results['tool_usage'].get(tool, 0) + 1
                
                # Track category performance
                cat = prompt_obj.category
                if cat not in results['category_performance']:
                    results['category_performance'][cat] = {'total': 0, 'success': 0}
                results['category_performance'][cat]['total'] += 1
                if prompt_obj.success:
                    results['category_performance'][cat]['success'] += 1
                
                # Track complexity performance
                comp = prompt_obj.complexity
                if comp not in results['complexity_performance']:
                    results['complexity_performance'][comp] = {'total': 0, 'success': 0}
                results['complexity_performance'][comp]['total'] += 1
                if prompt_obj.success:
                    results['complexity_performance'][comp]['success'] += 1
                    
            except Exception as e:
                print(f"❌ Error testing prompt: {e}")
                prompt_obj.tested = True
                prompt_obj.success = False
                results['failed'] += 1
        
        if results['total_tested'] > 0:
            results['avg_response_length'] = total_length / results['total_tested']
        
        return results
    
    def _evaluate_response_quality(self, response: str, prompt_obj: GeneratedPrompt) -> float:
        """Evaluate response quality based on various criteria"""
        score = 0.0
        
        # Length check (reasonable response)
        if 50 <= len(response) <= 5000:
            score += 0.2
        
        # Content relevance checks
        baseball_keywords = ['player', 'team', 'stat', 'game', 'season', 'average', 'home run', 'RBI']
        keyword_count = sum(1 for keyword in baseball_keywords if keyword.lower() in response.lower())
        score += min(keyword_count * 0.1, 0.3)
        
        # Error checks (negative indicators)
        error_indicators = ['error', 'failed', 'unable', 'not found', 'exception']
        if any(error in response.lower() for error in error_indicators):
            score -= 0.3
        
        # Positive indicators
        positive_indicators = ['statistics', 'performance', 'analysis', 'data', 'results']
        positive_count = sum(1 for indicator in positive_indicators if indicator.lower() in response.lower())
        score += min(positive_count * 0.1, 0.2)
        
        # Structure indicators (lists, formatting)
        if any(indicator in response for indicator in ['1.', '•', '-', '**', '🏆']):
            score += 0.2
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def generate_and_test_autonomous_batch(self, batch_size: int = 25, test_limit: int = 10) -> Dict[str, Any]:
        """Generate and test a batch of prompts autonomously"""
        print(f"\n🤖 AUTONOMOUS PROMPT GENERATION STARTED")
        print(f"Generating {batch_size} prompts, testing {test_limit}...")
        
        # Generate prompts
        new_prompts = self.generate_prompt_variations(batch_size)
        print(f"✅ Generated {len(new_prompts)} new prompts")
        
        # Test them
        results = self.test_prompt_batch(new_prompts, test_limit)
        
        # Save results
        self.save_prompt_library()
        
        return {
            'generation_results': {
                'prompts_generated': len(new_prompts),
                'categories': list(set(p.category for p in new_prompts)),
                'complexities': list(set(p.complexity for p in new_prompts))
            },
            'test_results': results,
            'best_prompts': self.get_best_prompts(5),
            'recommendations': self.generate_recommendations(results)
        }
    
    def get_best_prompts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the best performing prompts"""
        tested_prompts = [p for p in self.generated_prompts if p.tested and p.success]
        tested_prompts.sort(key=lambda x: x.response_quality, reverse=True)
        
        return [
            {
                'prompt': p.prompt,
                'category': p.category,
                'complexity': p.complexity,
                'quality_score': p.response_quality,
                'response_length': p.response_length
            }
            for p in tested_prompts[:limit]
        ]
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Category performance analysis
        if results['category_performance']:
            best_category = max(results['category_performance'].items(), 
                              key=lambda x: x[1]['success'] / max(x[1]['total'], 1))
            worst_category = min(results['category_performance'].items(), 
                               key=lambda x: x[1]['success'] / max(x[1]['total'], 1))
            
            recommendations.append(f"🎯 Best performing category: {best_category[0]}")
            recommendations.append(f"⚠️ Needs improvement: {worst_category[0]}")
        
        # Success rate analysis
        if results['total_tested'] > 0:
            success_rate = results['successful'] / results['total_tested']
            if success_rate > 0.8:
                recommendations.append("🌟 Excellent success rate! Consider increasing complexity.")
            elif success_rate < 0.5:
                recommendations.append("🔧 Low success rate. Focus on simpler templates.")
        
        # Tool usage analysis
        if results['tool_usage']:
            most_used_tool = max(results['tool_usage'].items(), key=lambda x: x[1])
            recommendations.append(f"🔨 Most utilized tool: {most_used_tool[0]}")
        
        return recommendations
    
    def save_prompt_library(self, filename: str = "autonomous_prompt_library.json"):
        """Save the generated prompt library"""
        library_data = {
            'metadata': {
                'total_prompts': len(self.generated_prompts),
                'tested_prompts': len([p for p in self.generated_prompts if p.tested]),
                'successful_prompts': len([p for p in self.generated_prompts if p.success]),
                'generated_at': datetime.now().isoformat(),
                'templates_count': len(self.prompt_templates)
            },
            'prompts': [asdict(p) for p in self.generated_prompts],
            'templates': [asdict(t) for t in self.prompt_templates]
        }
        
        with open(filename, 'w') as f:
            json.dump(library_data, f, indent=2)
        
        print(f"💾 Prompt library saved to {filename}")
        return filename
    
    def load_prompt_library(self, filename: str = "autonomous_prompt_library.json"):
        """Load an existing prompt library"""
        try:
            with open(filename, 'r') as f:
                library_data = json.load(f)
            
            # Reconstruct prompt objects
            self.generated_prompts = [GeneratedPrompt(**p) for p in library_data['prompts']]
            
            print(f"📖 Loaded {len(self.generated_prompts)} prompts from {filename}")
            return True
        except FileNotFoundError:
            print(f"📁 No existing library found at {filename}")
            return False
    
    def continuous_generation_mode(self, cycles: int = 5, prompts_per_cycle: int = 20, tests_per_cycle: int = 8):
        """Run continuous autonomous prompt generation"""
        print(f"\n🔄 CONTINUOUS AUTONOMOUS MODE STARTED")
        print(f"Running {cycles} cycles, {prompts_per_cycle} prompts per cycle, testing {tests_per_cycle} each")
        
        all_results = []
        
        for cycle in range(cycles):
            print(f"\n🔄 CYCLE {cycle + 1}/{cycles}")
            
            cycle_results = self.generate_and_test_autonomous_batch(
                batch_size=prompts_per_cycle,
                test_limit=tests_per_cycle
            )
            
            all_results.append({
                'cycle': cycle + 1,
                'timestamp': datetime.now().isoformat(),
                'results': cycle_results
            })
            
            print(f"✅ Cycle {cycle + 1} complete")
            
            # Print cycle summary
            test_results = cycle_results['test_results']
            print(f"   Success Rate: {test_results['successful']}/{test_results['total_tested']}")
            print(f"   Avg Response Length: {test_results['avg_response_length']:.0f} chars")
        
        # Final summary
        total_generated = sum(r['results']['generation_results']['prompts_generated'] for r in all_results)
        total_tested = sum(r['results']['test_results']['total_tested'] for r in all_results)
        total_successful = sum(r['results']['test_results']['successful'] for r in all_results)
        
        print(f"\n🎉 CONTINUOUS GENERATION COMPLETE")
        print(f"📊 Final Stats:")
        print(f"   Total Prompts Generated: {total_generated}")
        print(f"   Total Prompts Tested: {total_tested}")
        print(f"   Total Successful: {total_successful}")
        print(f"   Overall Success Rate: {total_successful/max(total_tested,1)*100:.1f}%")
        
        return {
            'summary': {
                'cycles_completed': cycles,
                'total_generated': total_generated,
                'total_tested': total_tested,
                'total_successful': total_successful,
                'overall_success_rate': total_successful/max(total_tested,1)
            },
            'cycle_results': all_results,
            'final_library_stats': {
                'total_prompts': len(self.generated_prompts),
                'best_prompts': self.get_best_prompts(10)
            }
        }

def main():
    """Run autonomous prompt generation"""
    generator = AutonomousPromptGenerator()
    
    # Load existing library if available
    generator.load_prompt_library()
    
    # Run continuous generation
    results = generator.continuous_generation_mode(
        cycles=3,
        prompts_per_cycle=15,
        tests_per_cycle=5
    )
    
    print(f"\n🎯 BEST PROMPTS DISCOVERED:")
    for i, prompt in enumerate(results['final_library_stats']['best_prompts'][:5], 1):
        print(f"{i}. {prompt['prompt']} (Quality: {prompt['quality_score']:.2f})")

if __name__ == "__main__":
    main()
