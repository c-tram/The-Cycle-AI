"""
Query-Driven Scraper  
Intelligently navigates the website based on the user's specific question
Actually USES the filters and views like a human would!
"""

import asyncio
from playwright.async_api import async_playwright
import re
from typing import Dict, List, Optional

class QueryDrivenScraper:
    """Scraper that understands queries and uses website features accordingly"""
    
    def __init__(self):
        self.browser = None
        self.context = None
        
    async def start(self):
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        
    async def stop(self):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    async def scrape_for_query(self, query: str, base_url: str = "https://thecycle.online/players") -> Dict:
        """Scrape data specifically for the given query"""
        
        if not self.browser:
            await self.start()
            
        page = await self.context.new_page()
        
        try:
            print(f"🎯 Query-driven scraping for: '{query}'")
            
            # Parse what the user wants
            query_intent = self.parse_query_intent(query)
            print(f"🧠 Query intent: {query_intent}")
            
            # Load the page
            await page.goto(base_url, wait_until='networkidle')
            await asyncio.sleep(3)  # Let it load
            
            # Apply filters and settings based on query
            modifications_made = await self.apply_query_specific_filters(page, query_intent)
            
            # Get the final data
            html_content = await page.content()
            
            result = {
                "html_content": html_content,
                "query_intent": query_intent,
                "modifications_made": modifications_made,
                "data_found": True
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Query-driven scraping failed: {e}")
            return {"error": str(e), "data_found": False}
            
        finally:
            await page.close()
    
    def parse_query_intent(self, query: str) -> Dict:
        """Parse what the user is actually asking for"""
        query_lower = query.lower()
        
        intent = {
            "team_filter": None,
            "stat_focus": None,
            "sort_direction": None,
            "filters_needed": [],
            "actions": []
        }
        
        # Extract team
        team_patterns = [
            r'\b(hou|houston|astros)\b',
            r'\b(nyy|yankees|new york)\b', 
            r'\b(lad|dodgers|los angeles)\b',
            r'\b(bos|red sox|boston)\b'
        ]
        
        for pattern in team_patterns:
            match = re.search(pattern, query_lower)
            if match:
                team_text = match.group(1)
                if team_text in ['hou', 'houston', 'astros']:
                    intent["team_filter"] = "HOU"
                elif team_text in ['nyy', 'yankees', 'new york']:
                    intent["team_filter"] = "NYY"
                elif team_text in ['lad', 'dodgers', 'los angeles']:
                    intent["team_filter"] = "LAD"
                elif team_text in ['bos', 'red sox', 'boston']:
                    intent["team_filter"] = "BOS"
                break
        
        # Extract stat focus
        if 'cvr' in query_lower:
            intent["stat_focus"] = "CVR"
        elif 'avg' in query_lower or 'batting average' in query_lower:
            intent["stat_focus"] = "AVG"
        elif 'ops' in query_lower:
            intent["stat_focus"] = "OPS"
        elif 'hr' in query_lower or 'home run' in query_lower:
            intent["stat_focus"] = "HR"
        
        # Extract sort direction
        if any(word in query_lower for word in ['highest', 'most', 'best', 'top']):
            intent["sort_direction"] = "DESC"
        elif any(word in query_lower for word in ['lowest', 'least', 'worst', 'bottom']):
            intent["sort_direction"] = "ASC"
        
        # Determine actions needed
        if intent["team_filter"]:
            intent["actions"].append(f"filter_team_{intent['team_filter']}")
        if intent["stat_focus"]:
            intent["actions"].append(f"enable_stat_{intent['stat_focus']}")
            if intent["sort_direction"]:
                intent["actions"].append(f"sort_by_{intent['stat_focus']}_{intent['sort_direction']}")
        
        return intent
    
    async def apply_query_specific_filters(self, page, query_intent: Dict) -> List[str]:
        """Apply website filters based on the parsed query"""
        
        modifications = []
        
        # 1. Filter by team if needed
        if query_intent.get("team_filter"):
            team = query_intent["team_filter"]
            success = await self.filter_by_team(page, team)
            if success:
                modifications.append(f"Filtered to {team} players")
        
        # 2. Enable specific stat column if needed
        if query_intent.get("stat_focus"):
            stat = query_intent["stat_focus"]
            success = await self.enable_stat_column(page, stat)
            if success:
                modifications.append(f"Enabled {stat} column")
        
        # 3. Sort by stat if needed
        if query_intent.get("stat_focus") and query_intent.get("sort_direction"):
            stat = query_intent["stat_focus"]
            direction = query_intent["sort_direction"]
            success = await self.sort_by_stat(page, stat, direction)
            if success:
                modifications.append(f"Sorted by {stat} {direction}")
        
        return modifications
    
    async def filter_by_team(self, page, team: str) -> bool:
        """Try to filter by a specific team"""
        
        team_filter_selectors = [
            f'select[data-testid*="team"]',
            f'select[name*="team"]',
            f'.team-filter',
            f'[data-filter="team"]',
            f'select:has(option[value*="{team}"])',
        ]
        
        for selector in team_filter_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    # Try to select the team
                    await element.select_option(value=team)
                    await page.wait_for_timeout(2000)  # Wait for filter to apply
                    print(f"✅ Successfully filtered to {team}")
                    return True
            except:
                continue
        
        # Try clicking team buttons/tabs
        team_button_selectors = [
            f'button:has-text("{team}")',
            f'.team-tab:has-text("{team}")',
            f'[data-team="{team}"]'
        ]
        
        for selector in team_button_selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    await page.wait_for_timeout(2000)
                    print(f"✅ Clicked {team} filter")
                    return True
            except:
                continue
        
        print(f"⚠️ Could not filter to {team}")
        return False
    
    async def enable_stat_column(self, page, stat: str) -> bool:
        """Try to enable a specific stat column"""
        
        # Look for column toggles, settings, or "more stats" buttons
        column_selectors = [
            f'input[type="checkbox"][value*="{stat}"]',
            f'button:has-text("{stat}")',
            f'button:has-text("More Stats")',
            f'button:has-text("Show All")',
            f'.column-toggle:has-text("{stat}")',
        ]
        
        for selector in column_selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    await page.wait_for_timeout(2000)
                    print(f"✅ Enabled {stat} column")
                    return True
            except:
                continue
        
        print(f"⚠️ Could not enable {stat} column")
        return False
    
    async def sort_by_stat(self, page, stat: str, direction: str) -> bool:
        """Try to sort by a specific stat"""
        
        # Look for sortable column headers
        sort_selectors = [
            f'th:has-text("{stat}")',
            f'.column-header:has-text("{stat}")',
            f'[data-sort="{stat}"]',
            f'button:has-text("{stat}")'
        ]
        
        for selector in sort_selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    
                    # If we want ascending and it's currently descending, click again
                    if direction == "ASC":
                        await page.wait_for_timeout(500)
                        await element.click()
                    
                    await page.wait_for_timeout(2000)
                    print(f"✅ Sorted by {stat} {direction}")
                    return True
            except:
                continue
        
        print(f"⚠️ Could not sort by {stat}")
        return False

# Synchronous wrapper
class SyncQueryDrivenScraper:
    def __init__(self):
        self.scraper = QueryDrivenScraper()
    
    def scrape_for_query(self, query: str) -> Dict:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run():
                await self.scraper.start()
                try:
                    return await self.scraper.scrape_for_query(query)
                finally:
                    await self.scraper.stop()
            
            result = loop.run_until_complete(run())
            loop.close()
            return result
        except Exception as e:
            return {"error": str(e), "data_found": False}

# Test
if __name__ == "__main__":
    scraper = SyncQueryDrivenScraper()
    result = scraper.scrape_for_query("which HOU player has the highest CVR?")
    
    print(f"Success: {result.get('data_found', False)}")
    print(f"Modifications: {result.get('modifications_made', [])}")
    print(f"Intent: {result.get('query_intent', {})}")
