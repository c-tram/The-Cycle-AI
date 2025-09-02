"""
JavaScript-Aware Web Scraper using Playwright
Handles dynamic content and React/SPA applications
"""

import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional
import json
import time

class JavaScriptScraper:
    """Scrapes dynamic JavaScript websites with full rendering"""
    
    def __init__(self):
        self.browser = None
        self.context = None
        
    async def start(self):
        """Initialize browser"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        
    async def stop(self):
        """Clean up browser"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
            
    async def scrape_thecycle_players(self, wait_time=5, explore_all_data=True, endpoint="players") -> Dict:
        """
        Scrape player data from thecycle.online with full JavaScript rendering
        and interactive exploration of filters/buttons
        
        Args:
            wait_time: Time to wait for JavaScript to render
            explore_all_data: Whether to explore interactive features
            endpoint: The endpoint to scrape ('players', 'players/pitching', 'players/batting')
        """
        if not self.browser:
            await self.start()
            
        page = await self.context.new_page()
        
        try:
            # Use the specified endpoint
            url = f'https://thecycle.online/{endpoint}'
            print(f"🌐 Loading {url}...")
            await page.goto(url, wait_until='networkidle')
            
            # Wait for dynamic content to load
            print(f"⏳ Waiting {wait_time}s for JavaScript to render...")
            await asyncio.sleep(wait_time)
            
            # Try to wait for specific elements that indicate data has loaded
            try:
                await page.wait_for_selector('[data-testid*="player"], .player-row, table, [class*="table"], .stats-table', timeout=10000)
                print("✅ Found data elements!")
            except:
                print("⚠️ No specific data selectors found, proceeding with general content...")
            
            if explore_all_data:
                await self.explore_interactive_features(page)
            
            # Get the fully rendered HTML
            html_content = await page.content()
            print(f"📄 Final rendered HTML length: {len(html_content)} characters")
            
            # Extract data structures
            result = await self.extract_player_data(page, html_content)
            
            return result
            
        except Exception as e:
            print(f"❌ Error scraping: {e}")
            return {"error": str(e), "html_content": ""}
            
        finally:
            await page.close()
    
    async def explore_interactive_features(self, page):
        """Explore and interact with buttons, filters, and other UI elements"""
        
        print("🔍 Exploring interactive features...")
        
        # Look for common UI patterns
        selectors_to_try = [
            # Buttons
            'button[contains(text(), "Filter")]',
            'button[contains(text(), "Show")]',
            'button[contains(text(), "More")]',
            'button[contains(text(), "All")]',
            'button[contains(text(), "CVR")]',
            '.btn, .button',
            '[role="button"]',
            
            # Dropdowns and selects
            'select',
            '.dropdown, .select',
            '[data-testid*="filter"]',
            '[data-testid*="sort"]',
            
            # Tabs
            '.tab, .nav-tab',
            '[role="tab"]',
            
            # Pagination
            '.pagination',
            '[aria-label*="page"]',
            '.next, .prev',
            
            # Column toggles
            '[data-column]',
            '.column-toggle',
            
            # Stats toggles
            '.stats-toggle',
            '[data-stat]'
        ]
        
        interactive_elements = []
        for selector in selectors_to_try:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    text_content = await element.text_content()
                    is_visible = await element.is_visible()
                    
                    if is_visible and text_content:
                        interactive_elements.append({
                            'element': element,
                            'text': text_content.strip(),
                            'selector': selector
                        })
            except:
                continue
        
        print(f"🎛️ Found {len(interactive_elements)} interactive elements")
        
        # Try to find and click CVR or comprehensive stats
        cvr_found = False
        for item in interactive_elements:
            text = item['text'].lower()
            if any(keyword in text for keyword in ['cvr', 'comprehensive', 'value', 'rating', 'advanced', 'more stats']):
                print(f"🎯 Trying to click: '{item['text']}'")
                try:
                    await item['element'].click()
                    await page.wait_for_timeout(2000)  # Wait for content to load
                    cvr_found = True
                    print("✅ Successfully clicked CVR-related element!")
                    break
                except Exception as e:
                    print(f"❌ Failed to click: {e}")
        
        # Try to expand all columns or show more data
        if not cvr_found:
            for item in interactive_elements:
                text = item['text'].lower()
                if any(keyword in text for keyword in ['show all', 'expand', 'more columns', 'all stats']):
                    print(f"📊 Trying to show more data: '{item['text']}'")
                    try:
                        await item['element'].click()
                        await page.wait_for_timeout(2000)
                        print("✅ Successfully expanded data view!")
                        break
                    except Exception as e:
                        print(f"❌ Failed to expand: {e}")
        
        # Look for dropdown filters that might show CVR
        try:
            dropdowns = await page.query_selector_all('select, .dropdown-toggle, [role="combobox"]')
            for dropdown in dropdowns:
                try:
                    # Click to open dropdown
                    await dropdown.click()
                    await page.wait_for_timeout(1000)
                    
                    # Look for CVR options
                    options = await page.query_selector_all('option, .dropdown-item, .menu-item')
                    for option in options:
                        option_text = await option.text_content()
                        if option_text and 'cvr' in option_text.lower():
                            print(f"🎯 Found CVR option: '{option_text}'")
                            await option.click()
                            await page.wait_for_timeout(2000)
                            cvr_found = True
                            break
                    
                    if cvr_found:
                        break
                        
                except Exception as e:
                    print(f"⚠️ Dropdown interaction failed: {e}")
                    
        except Exception as e:
            print(f"⚠️ Dropdown exploration failed: {e}")
        
        # Try pagination to get more players
        try:
            next_buttons = await page.query_selector_all('.next, [aria-label*="next"], .pagination-next')
            for next_button in next_buttons:
                if await next_button.is_visible():
                    print("📄 Found pagination, getting more players...")
                    await next_button.click()
                    await page.wait_for_timeout(2000)
                    break
        except Exception as e:
            print(f"⚠️ Pagination failed: {e}")
        
        print("🔍 Interactive exploration complete!")
    
    async def extract_player_data(self, page, html_content: str) -> Dict:
        """Extract structured player data from the rendered page"""
        
        result = {
            "html_content": html_content,
            "tables": [],
            "data_found": False,
            "extraction_method": "unknown"
        }
        
        # Method 1: Look for HTML tables
        soup = BeautifulSoup(html_content, 'html.parser')
        tables = soup.find_all('table')
        
        if tables:
            print(f"📊 Found {len(tables)} HTML tables")
            try:
                df_tables = pd.read_html(html_content)
                for i, table in enumerate(df_tables):
                    result["tables"].append({
                        "index": i,
                        "columns": list(table.columns),
                        "rows": len(table),
                        "data": table.to_dict('records')[:50]  # Limit to 50 rows for performance
                    })
                result["data_found"] = True
                result["extraction_method"] = "html_tables"
                return result
            except Exception as e:
                print(f"⚠️ Error parsing HTML tables: {e}")
        
        # Method 2: Look for data in JavaScript variables
        try:
            # Extract data from window objects or embedded JSON
            js_data = await page.evaluate("""
                () => {
                    // Look for common data patterns
                    const data = {};
                    
                    // Check for players data in window
                    if (window.playersData) data.playersData = window.playersData;
                    if (window.__INITIAL_STATE__) data.initialState = window.__INITIAL_STATE__;
                    if (window.APP_DATA) data.appData = window.APP_DATA;
                    
                    // Look for React components data
                    const reactElements = document.querySelectorAll('[data-reactroot], [data-react-props]');
                    data.reactElementsCount = reactElements.length;
                    
                    // Try to find player rows or data
                    const playerRows = document.querySelectorAll('[data-testid*="player"], .player-row, [class*="player"], tr');
                    if (playerRows.length > 0) {
                        data.playerElements = Array.from(playerRows).slice(0, 20).map(el => {
                            const cells = el.querySelectorAll('td, .cell, [class*="cell"]');
                            return {
                                text: el.textContent?.trim(),
                                classes: el.className,
                                cellCount: cells.length,
                                cells: Array.from(cells).map(cell => cell.textContent?.trim()).filter(Boolean)
                            };
                        });
                    }
                    
                    // Look for column headers to understand what stats are available
                    const headers = document.querySelectorAll('th, .header, [class*="header"], .column-header');
                    if (headers.length > 0) {
                        data.columnHeaders = Array.from(headers).map(header => ({
                            text: header.textContent?.trim(),
                            classes: header.className
                        })).filter(h => h.text);
                    }
                    
                    // Look for CVR specifically
                    const cvrElements = document.querySelectorAll('*');
                    let cvrFound = false;
                    for (let el of cvrElements) {
                        if (el.textContent?.includes('CVR') || el.textContent?.includes('Comprehensive')) {
                            cvrFound = true;
                            break;
                        }
                    }
                    data.cvrPresent = cvrFound;
                    
                    // Count visible interactive elements
                    const buttons = document.querySelectorAll('button:not([disabled])');
                    const selects = document.querySelectorAll('select');
                    const tabs = document.querySelectorAll('.tab, [role="tab"]');
                    
                    data.interactiveElements = {
                        buttons: buttons.length,
                        selects: selects.length,
                        tabs: tabs.length,
                        buttonTexts: Array.from(buttons).slice(0, 10).map(b => b.textContent?.trim()).filter(Boolean)
                    };
                    
                    // Look for any tables or grids
                    const dataElements = document.querySelectorAll('table, [role="grid"], [class*="table"], [class*="grid"]');
                    if (dataElements.length > 0) {
                        data.dataElements = Array.from(dataElements).map(el => ({
                            tagName: el.tagName,
                            className: el.className,
                            textContent: el.textContent?.trim().substring(0, 500)
                        }));
                    }
                    
                    return data;
                }
            """)
            
            if js_data and any(js_data.values()):
                print("🔍 Found JavaScript data!")
                result["js_data"] = js_data
                result["data_found"] = True
                result["extraction_method"] = "javascript_extraction"
                
                # Try to structure the data
                if "playerElements" in js_data and js_data["playerElements"]:
                    players = []
                    for element in js_data["playerElements"]:
                        # Basic text parsing - you'd make this more sophisticated
                        text = element.get("text", "")
                        if text and len(text.split()) > 3:  # Has meaningful content
                            players.append({"raw_text": text})
                    
                    if players:
                        result["tables"] = [{
                            "index": 0,
                            "columns": ["raw_text"],
                            "rows": len(players),
                            "data": players
                        }]
                
        except Exception as e:
            print(f"⚠️ Error extracting JavaScript data: {e}")
        
        # Method 3: Text-based extraction
        if not result["data_found"]:
            print("📝 Falling back to text extraction...")
            text_content = soup.get_text()
            if len(text_content) > 1000:  # Has substantial content
                result["text_content"] = text_content[:5000]  # First 5000 chars
                result["data_found"] = True
                result["extraction_method"] = "text_fallback"
        
        return result

# Synchronous wrapper for use in Streamlit
class SyncJavaScriptScraper:
    """Synchronous wrapper for the async scraper"""
    
    def __init__(self):
        self.scraper = JavaScriptScraper()
    
    def scrape_players(self, wait_time=5, explore_all_data=True, endpoint="players") -> Dict:
        """Synchronously scrape player data with full exploration"""
        try:
            # Run the async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def run_scraper():
                await self.scraper.start()
                try:
                    result = await self.scraper.scrape_thecycle_players(wait_time, explore_all_data, endpoint)
                    return result
                finally:
                    await self.scraper.stop()
            
            result = loop.run_until_complete(run_scraper())
            loop.close()
            return result
            
        except Exception as e:
            return {"error": str(e), "data_found": False}

# Test function
async def test_scraper():
    """Test the scraper"""
    scraper = JavaScriptScraper()
    await scraper.start()
    
    try:
        result = await scraper.scrape_thecycle_players()
        print(f"Data found: {result['data_found']}")
        print(f"Method: {result['extraction_method']}")
        print(f"Tables: {len(result.get('tables', []))}")
        
        if result.get('tables'):
            for i, table in enumerate(result['tables']):
                print(f"Table {i}: {table['rows']} rows, columns: {table['columns']}")
                
    finally:
        await scraper.stop()

if __name__ == "__main__":
    print("🚀 Testing JavaScript scraper...")
    asyncio.run(test_scraper())
