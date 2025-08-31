import re
from exa_py import Exa
from prompts import SEARCH_QUERIES_PROMPT
from config import EXA_API_KEY

NUM_SEARCH_QUERIES = 5


async def generate_search_queries(question: str, num_queries: int = 3) -> tuple[list[str], str]:
    """
    Use OpenAI to generate optimized search queries and suggest the most appropriate start date for the search.
    Returns (queries, start_date)
    """
    from llm_calls import call_openAI
        
    prompt = SEARCH_QUERIES_PROMPT.format(num_queries=num_queries, question=question)
    
    try:
        response = await call_openAI(prompt, model="gpt-4.1", temperature=0.1)
        
        print(f"OpenAI response for search query generation:\n{response}\n" + "="*50)
        
        # Parse the response
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        
        start_date = "2023-01-01T00:00:00.000Z"  # default
        queries = []
        
        for line in lines:
            if line.startswith("START_DATE:"):
                start_date = line.replace("START_DATE:", "").strip()
            elif line.startswith("QUERY_"):
                query = line.split(":", 1)[1].strip() if ":" in line else line
                queries.append(query)
        
        # Fallback if parsing fails
        if not queries:
            queries = [question]
            
        return queries[:num_queries] if len(queries) >= num_queries else queries, start_date
        
    except Exception as e:
        print(f"Error generating search queries: {e}")
        return [question], "2023-01-01T00:00:00.000Z"


def call_exa_search(query: str, start_published_date: str = "2023-01-01T00:00:00.000Z") -> dict:
    """
    Perform a single Exa search with the given query and start date.
    """
    exa = Exa(api_key=EXA_API_KEY)
    response = exa.search_and_contents(
        query=query,
        start_published_date=start_published_date,
        num_results=5,
        category="news",
        type = "auto",
        summary = {
          "query": "You are tasked with creating a clean, concise summary by combining and cleaning multiple content sources from a web article. This summary will be used by a professional forecaster to make predictions about future events. Focus on factual information, key points, and relevant details that would be valuable for forecasting. If the content is not substantially relevant for forecasting purposes or is mostly noise, return \"No relevant content found.\""
        }
    )
    return response


async def run_exa_research(question: str) -> str:
    """
    Use OpenAI to generate optimized search queries, then use Exa Search API to find relevant content.
    Ensures no duplicate results based on URL.
    """
    if not EXA_API_KEY:
        return "Exa API key not provided."
    
    # Generate optimized search queries and get suggested start date using OpenAI
    search_queries, start_date = await generate_search_queries(question, num_queries=NUM_SEARCH_QUERIES)
    
    print(f"Generated {len(search_queries)} search queries with start date: {start_date}")
    
    # Track seen URLs to avoid duplicates
    seen_urls = set()
    all_unique_results = []
    
    try:
        # Collect all results from all queries
        for i, query in enumerate(search_queries, 1):
            try:
                search_data = call_exa_search(query, start_date)
                
                # Handle SearchResponse object from exa-py
                if hasattr(search_data, 'results'):
                    results = search_data.results
                elif hasattr(search_data, '__iter__'):
                    results = search_data
                else:
                    print(f"Unexpected response format for query {i}: {type(search_data)}")
                    continue
                
                if not results:
                    print(f"No results found for query {i}.")
                    continue
                
                # Filter out duplicates and add unique results
                for result in results:
                    # Convert result object to dict if needed
                    if hasattr(result, '__dict__'):
                        result_dict = result.__dict__
                    elif hasattr(result, 'url'):
                        result_dict = {
                            'url': result.url,
                            'title': getattr(result, 'title', ''),
                            'text': getattr(result, 'text', ''),
                            'publishedDate': getattr(result, 'published_date', ''),
                            'summary': getattr(result, 'summary', ''),
                            'highlights': getattr(result, 'highlights', [])
                        }
                    else:
                        result_dict = result
                    
                    url = result_dict.get('url', '')
                    summary = result_dict.get('summary', '')
                    
                    # Filter out results with no relevant content and avoid duplicates
                    # Use regex to catch variations of "no relevant content found"
                    if url and url not in seen_urls and not re.search(r'no\s+relevant\s+content\s+found', summary.lower()):
                        seen_urls.add(url)
                        result_dict['source_query'] = query
                        result_dict['query_number'] = i
                        all_unique_results.append(result_dict)
                        
            except Exception as e:
                print(f"Error processing query {i} ('{query}'): {str(e)}")
                continue
        
        # Sort results by published date (newest first) if available
        def sort_key(result):
            pub_date = result.get('publishedDate', '')
            return pub_date if pub_date else '1900-01-01'
        
        all_unique_results.sort(key=sort_key, reverse=True)
        
        # Format the deduplicated results
        formatted_results = f"Here are {len(all_unique_results)} unique search results (duplicates removed):\n\n"

        for i, result in enumerate(all_unique_results[:20], 1):  # Limit to top 20 unique results
            # formatted_results += f"[Source {i}] (from query: '{result['source_query']}'):\n"
            formatted_results += f"[Source {i}]:\n"
            formatted_results += f"Title: {result.get('title', 'No title')}\n"
            formatted_results += f"URL: {result.get('url', 'No URL')}\n"

            if result.get('publishedDate'):
                formatted_results += f"Published: {result['publishedDate']}\n"
            
            if result.get('summary'):
                formatted_results += f"Summary: {result['summary']}\n"
          
            formatted_results += "\n"
        
        return formatted_results
        
    except Exception as e:
        return f"Error in smart search: {str(e)}"
