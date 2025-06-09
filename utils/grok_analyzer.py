#!/usr/bin/env python3
"""
Grok API Integration for Advanced Analysis
"""
import os
import json
import requests
from typing import Dict, List, Optional
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GrokAnalyzer:
    def __init__(self):
        """Initialize the Grok analyzer"""
        self.api_key = os.getenv('GROK_API_KEY')
        self.api_url = "https://api.grok.ai/v1/analyze"  # Replace with actual API endpoint
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
    def analyze_clusters(self, cluster_topics: Dict[int, List[str]],
                        cluster_stats: Dict) -> Optional[Dict]:
        """Analyze clusters using Grok AI"""
        try:
            # Prepare the prompt
            prompt = self._create_analysis_prompt(cluster_topics, cluster_stats)
            
            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    'prompt': prompt,
                    'max_tokens': 1000,
                    'temperature': 0.7
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error in Grok analysis: {e}")
            return None
            
    def _create_analysis_prompt(self, cluster_topics: Dict[int, List[str]],
                              cluster_stats: Dict) -> str:
        """Create a detailed analysis prompt for Grok"""
        prompt = f"""Analyze the following job market clusters and provide insights:

Cluster Topics:
{json.dumps(cluster_topics, indent=2)}

Cluster Statistics:
{json.dumps(cluster_stats, indent=2)}

Please provide:
1. A detailed analysis of each cluster's characteristics
2. Key trends and patterns identified
3. Market insights and implications
4. Recommendations for job seekers and employers
5. Future market predictions based on the data

Format the response as a structured report with clear sections and bullet points.
"""
        return prompt
        
    def generate_insights(self, analysis: Dict) -> str:
        """Generate human-readable insights from Grok's analysis"""
        try:
            # Extract the analysis text
            analysis_text = analysis.get('text', '')
            
            # Format the insights
            insights = f"""# Job Market Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{analysis_text}

---
This analysis was generated using Grok AI's advanced natural language processing capabilities.
"""
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return "Error generating insights"
            
    def save_analysis(self, insights: str, output_path: str = 'reports/grok_analysis.md'):
        """Save the analysis to a markdown file"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(insights)
            logger.info(f"Analysis saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
            return False

def main():
    """Test the Grok analyzer"""
    # Load sample data
    with open('reports/kmeans_analysis.json', 'r') as f:
        cluster_topics = json.load(f)
        
    # Generate sample statistics
    cluster_stats = {
        0: {
            'size': 100,
            'avg_salary_min': 5000,
            'avg_salary_max': 8000,
            'top_locations': {'Casablanca': 30, 'Rabat': 20, 'Marrakech': 15},
            'top_categories': {'IT': 40, 'Marketing': 30, 'Sales': 20}
        }
    }
    
    # Initialize analyzer
    analyzer = GrokAnalyzer()
    
    # Get analysis
    analysis = analyzer.analyze_clusters(cluster_topics, cluster_stats)
    if analysis:
        # Generate insights
        insights = analyzer.generate_insights(analysis)
        
        # Save analysis
        analyzer.save_analysis(insights)

if __name__ == "__main__":
    main() 