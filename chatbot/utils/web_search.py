import requests
import logging
import json
from urllib.parse import quote, urlencode
from bs4 import BeautifulSoup

# Use the app's logger
logger = logging.getLogger('chatbot')

def web_search(query, max_results=3):
    logger.info(f"Starting web search for query: '{query}'")
    print(f"Starting web search for query: '{query}'")
    
    # Try multiple search methods in order of preference
    results = []
    
    # Method 1: Try serpapi.com if you have an API key (requires subscription)
    # Uncomment this section if you have a SerpAPI key
    try:
        serpapi_key = "a64064c88af685be70853dbb69756ac644bc9fde371a4771b5780f325762b5fc"  # Replace with your actual key
        if serpapi_key:
            logger.info("Trying SerpAPI search")
            params = {
                "engine": "google",
                "q": query,
                "api_key": serpapi_key,
                "num": max_results
            }
            response = requests.get("https://serpapi.com/search", params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "organic_results" in data:
                for result in data["organic_results"][:max_results]:
                    results.append({
                        "title": result.get("title", "Search Result"),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", "#")
                    })
                logger.info(f"SerpAPI search successful, found {len(results)} results")
                if results:
                    return format_results_as_html(results)
    except Exception as e:
        logger.error(f"SerpAPI search failed: {str(e)}")
        print(f"SerpAPI search failed: {str(e)}")
    
    # Method 2: Try Scraping DuckDuckGo HTML results (fallback method)
    try:
        logger.info("Trying DuckDuckGo HTML scraping")
        print("Trying DuckDuckGo HTML scraping")
        
        # Headers to mimic a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        # Query DuckDuckGo HTML version
        search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        logger.info(f"Making request to: {search_url}")
        print(f"Making request to: {search_url}")
        
        response = requests.get(search_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML response
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('div', class_='result')
        
        for i, result in enumerate(search_results):
            if i >= max_results:
                break
                
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')
            
            if title_elem and snippet_elem:
                title = title_elem.get_text().strip()
                snippet = snippet_elem.get_text().strip()
                link = title_elem.get('href', '#')
                
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "link": link
                })
        
        logger.info(f"DuckDuckGo HTML scraping found {len(results)} results")
        print(f"DuckDuckGo HTML scraping found {len(results)} results")
    except Exception as e:
        logger.error(f"DuckDuckGo HTML scraping failed: {str(e)}")
        print(f"DuckDuckGo HTML scraping failed: {str(e)}")
    
    # Method 3: Try Wikipedia API search
    if len(results) < max_results:
        try:
            logger.info("Trying Wikipedia API search")
            print("Trying Wikipedia API search")
            
            # Get search results from Wikipedia
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote(query)}&format=json&utf8=1"
            response = requests.get(wiki_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "query" in data and "search" in data["query"]:
                for item in data["query"]["search"]:
                    title = item.get("title", "Wikipedia Result")
                    snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text()
                    page_id = item.get("pageid", "")
                    link = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    
                    results.append({
                        "title": f"Wikipedia: {title}",
                        "snippet": snippet,
                        "link": link
                    })
                    
                    if len(results) >= max_results:
                        break
                        
                logger.info(f"Wikipedia API search found {len(data['query']['search'])} results")
                print(f"Wikipedia API search found {len(data['query']['search'])} results")
        except Exception as e:
            logger.error(f"Wikipedia API search failed: {str(e)}")
            print(f"Wikipedia API search failed: {str(e)}")
    
    # Method 4: Try crypto price API for cryptocurrency queries
    if "price" in query.lower() and any(crypto in query.lower() for crypto in ["bitcoin", "ethereum", "btc", "eth", "crypto"]):
        try:
            logger.info("Detected cryptocurrency price query, trying CoinGecko API")
            print("Detected cryptocurrency price query, trying CoinGecko API")
            
            crypto_map = {
                "bitcoin": "bitcoin",
                "btc": "bitcoin", 
                "ethereum": "ethereum",
                "eth": "ethereum"
            }
            
            # Extract the cryptocurrency from the query
            crypto_id = None
            for key, value in crypto_map.items():
                if key in query.lower():
                    crypto_id = value
                    break
            
            if crypto_id:
                # Get price data from CoinGecko
                crypto_url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_id}&vs_currencies=usd&include_24hr_change=true&include_last_updated_at=true"
                response = requests.get(crypto_url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if crypto_id in data:
                    price_usd = data[crypto_id].get("usd", "Unknown")
                    change_24h = data[crypto_id].get("usd_24h_change", "Unknown")
                    updated_at = data[crypto_id].get("last_updated_at", 0)
                    
                    # Format the change with + or - sign and percentage
                    if change_24h != "Unknown":
                        change_formatted = f"{'+' if change_24h >= 0 else ''}{change_24h:.2f}%"
                    else:
                        change_formatted = "Unknown"
                    
                    results.append({
                        "title": f"Current {crypto_id.capitalize()} Price",
                        "snippet": f"${price_usd:,} USD (24h change: {change_formatted})",
                        "link": f"https://www.coingecko.com/en/coins/{crypto_id}"
                    })
                    
                    logger.info(f"CoinGecko API found price data for {crypto_id}")
                    print(f"CoinGecko API found price data for {crypto_id}")
        except Exception as e:
            logger.error(f"CoinGecko API search failed: {str(e)}")
            print(f"CoinGecko API search failed: {str(e)}")
    
    # Check if we have any results
    if not results:
        logger.warning("No search results found from any method")
        print("No search results found from any method")
        return ""
    
    # Format results as HTML
    return format_results_as_html(results)

def format_results_as_html(results):
    """Format search results as HTML."""
    html = '<div class="web-search-results" style="font-family: Arial, sans-serif; line-height: 1.6; padding: 1em; background-color: #f9f9f9; border-radius: 5px; margin-bottom: 1em;">'
    html += "<h3>Web Search Results</h3><ul style='padding-left: 20px;'>"
    
    for res in results:
        title = res.get("title", "Result")
        snippet = res.get("snippet", "")
        link = res.get("link", "#")
        html += f'<li><strong>{title}</strong>: {snippet} <a href="{link}" target="_blank">Read More</a></li>'
    
    html += "</ul></div>"
    
    logger.info(f"Successfully formatted {len(results)} search results as HTML")
    print(f"Successfully formatted {len(results)} search results as HTML")
    
    return html