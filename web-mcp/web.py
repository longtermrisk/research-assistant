# mcp_web_server.py
import os
import serpapi
import requests
from markdownify import markdownify
from typing import List, Dict, Optional
import logging

from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Helper Functions (Adapted from original code) ---

def _format_search_results(results: List[Dict[str, str]]) -> str:
    """Formats the initial list of search results."""
    formatted = []
    for r in results:
        title = r.get('title', 'No Title')
        snippet = r.get('snippet', 'No Snippet')
        link = r.get('link', 'No Link')
        formatted.append(f"# {title}\n{snippet}\n{link}")
    return "\n\n".join(formatted)

def _markdown_browser(url: str) -> str:
    """Fetches a URL and converts its HTML content to markdown."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Fetch the HTML content with a timeout
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Decode HTML content, trying common encodings if needed
        response.encoding = response.apparent_encoding # Guess encoding
        html_content = response.text

        # Convert to markdown
        # You might want to configure markdownify further (e.g., strip=['script', 'style'])
        markdown_content = markdownify(html_content, heading_style="ATX")
        return markdown_content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return f"Error fetching URL {url}: {str(e)}"
    except Exception as e:
        logger.error(f"Error processing URL {url}: {e}")
        return f"Error processing URL {url}: {str(e)}"

# --- MCP Server Setup ---

# Explicitly list dependencies for installation/packaging
mcp = FastMCP(
    "WebSearchAgent",
    dependencies=["serpapi-python", "requests", "markdownify", "python-dotenv"]
)

# --- MCP Tools ---

@mcp.tool()
async def google_search(query: str, expand: int = 3) -> str:
    """
    Use Google to search the web. Returns the title, snippet, and link of
    top search results. Optionally expands the top 'expand' results by
    fetching and including their markdown content.
    """
    serp_api_key = os.getenv("SERP_API_KEY")
    if not serp_api_key:
        logger.error("SERP_API_KEY environment variable not set.")
        return "Error: Search API key not configured."

    logger.info(f"Performing Google search for query: '{query}', expanding top {expand} results.")

    try:
        # Initialize SerpAPI client
        client = serpapi.Client(api_key=serp_api_key)

        # Perform the search
        # Using Berlin, Germany location and en/us language as in the original code
        search = client.search(
            q=query,
            engine="google",
            location="Berlin, Germany",
            hl="en",
            gl="us"
        )
        search_results = search.as_dict()

        if "error" in search_results:
             logger.error(f"SerpAPI error: {search_results['error']}")
             return f"Search API Error: {search_results['error']}"

        organic_results = search_results.get("organic_results", [])
        if not organic_results:
            logger.info("No organic results found.")
            # Consider returning related questions or other result types if available
            answer_box = search_results.get("answer_box")
            if answer_box:
                 title = answer_box.get('title', '')
                 snippet = answer_box.get('snippet', answer_box.get('answer', ''))
                 return f"Answer Box:\n# {title}\n{snippet}"
            return "No search results found."

        # Extract relevant keys and format basic results
        keys = ["title", "snippet", "link"]
        filtered_results = [{k: i.get(k) for k in keys if i.get(k)} for i in organic_results]
        base_text = _format_search_results(filtered_results)

        # Expand top results by fetching markdown content
        expanded_content = []
        results_to_expand = filtered_results[:expand] # Use filtered results to ensure link exists

        logger.info(f"Attempting to expand {len(results_to_expand)} results.")
        for i, result_data in enumerate(results_to_expand):
            url = result_data.get('link')
            if url:
                logger.info(f"Expanding result {i+1}: Fetching markdown for {url}")
                markdown_text = _markdown_browser(url)
                # Add separator and source info
                expanded_content.append(f"\n\n" + "-"*50 + f"\n\nSource {i+1}: {url}\n\n{markdown_text}")
            else:
                 logger.warning(f"Skipping expansion for result {i+1} due to missing URL.")


        return base_text + "".join(expanded_content)

    except serpapi.SerpApiClientException as e:
         logger.error(f"SerpAPI client exception: {e}")
         return f"Error communicating with Search API: {str(e)}"
    except Exception as e:
        logger.exception(f"An unexpected error occurred during Google Search: {e}") # Log full traceback
        return f"An unexpected error occurred: {str(e)}"


@mcp.tool()
async def markdown_browser(url: str) -> str:
    """Opens a website URL and converts its HTML content to markdown."""
    logger.info(f"Fetching markdown for URL: {url}")
    # Reuse the internal helper function
    return _markdown_browser(url)


# --- Run the Server ---

if __name__ == "__main__":
    print("Starting MCP Web Search Agent Server...")
    # Ensure SERP_API_KEY is loaded before starting
    if not os.getenv("SERP_API_KEY"):
        print("\nWarning: SERP_API_KEY environment variable is not set.")
        print("Please set it or create a .env file with SERP_API_KEY=your_api_key\n")
    mcp.run() # Runs the server (e.g., via stdio by default)