"""
Advanced PDF Report Generator with AI Analysis
Integrates with Grok AI for intelligent insights
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import HexColor
import json
import os
import requests
from datetime import datetime
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TopicAnalysisReportGenerator:
    def __init__(self):
        """Initialize the advanced report generator"""
        self.grok_api_key = os.getenv('GROQ_API_KEY')
        self.report_data = None
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom styles for the report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        # Heading style
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        ))
        
        # Subheading style
        self.styles.add(ParagraphStyle(
            name='CustomSubheading',
            parent=self.styles['Heading2'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkgreen
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        # Bullet point style
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            leftIndent=20,
            bulletIndent=10
        ))

    def clean_ai_analysis_text(self, ai_analysis: str) -> str:
        """Clean AI analysis text for better readability and formatting"""
        # Don't strip formatting markers here - we'll handle them in the PDF creation
        return ai_analysis.strip()

    def generate_grok_prompt(self, report_data: Dict) -> str:
        """Generate comprehensive prompt for Grok AI analysis"""
        prompt = f"""
# Job Market Topic Analysis Report

## Analysis Overview
- Algorithm: {report_data['metadata']['algorithm']}
- Total Documents Analyzed: {report_data['metadata']['total_documents']}
- Number of Clusters Found: {report_data['metadata']['n_clusters']}
- Features Used: {report_data['metadata']['features_used']}

## Cluster Analysis Data

### Cluster Topics and Keywords:
"""
        
        for cluster_id, topics in report_data['cluster_topics'].items():
            prompt += f"\n**Cluster {cluster_id}** ({topics['size']} jobs):\n"
            prompt += f"Top Keywords: {', '.join(topics['keywords'][:10])}\n"
            
        prompt += "\n### Cluster Characteristics:\n"
        
        for cluster_id, chars in report_data['cluster_characteristics'].items():
            prompt += f"\n**Cluster {cluster_id} Profile:**\n"
            prompt += f"- Size: {chars['size']} jobs\n"
            prompt += f"- Top Sectors: {list(chars['top_sectors'].keys())[:3]}\n"
            prompt += f"- Top Locations: {list(chars['top_locations'].keys())[:3]}\n"
            prompt += f"- Source Distribution: {chars['source_distribution']}\n"
            
        prompt += """

## Analysis Request

Please provide a comprehensive analysis of this job market clustering data with the following sections:

### 1. Executive Summary
- Key findings and patterns identified
- Most significant market segments
- Overall market insights

### 2. Cluster Interpretation
- Meaningful names/labels for each cluster based on their characteristics
- Description of what each cluster represents in the job market
- Target audience and career relevance for each cluster

### 3. Market Trends Analysis
- Geographic distribution insights
- Sector concentration patterns
- Source platform analysis (MarocAnnonce vs Rekrute differences)

### 4. Strategic Recommendations
- For Job Seekers: Which clusters offer the best opportunities
- For Employers: Market positioning and talent acquisition strategies
- For Education/Training: Skills gap analysis and training recommendations

### 5. Future Predictions
- Emerging job market trends based on the clustering
- Potential growth areas
- Skills that will be in demand

Please make the analysis practical, actionable, and specifically relevant to the Moroccan job market context. Focus on insights that would be valuable for career counselors, HR professionals, and job seekers.

Format the response in clear sections with bullet points and actionable insights.
"""
        
        return prompt
        
    def call_grok_api(self, prompt: str) -> Optional[str]:
        """Call Grok AI API for analysis"""
        if not self.grok_api_key:
            logger.warning("Grok API key not found. Skipping AI analysis.")
            return None
            
        try:
            headers = {
                'Authorization': f'Bearer {self.grok_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                'messages': [
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                'temperature': 0.7,
                'max_completion_tokens': 2000
            }
            
            response = requests.post(
                'https://api.groq.com/openai/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                choices = response.json().get('choices', [])
                if choices:
                    return choices[0]['message']['content']
                return response.json().get('text', '')
            else:
                logger.error(f"Grok API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Grok API: {e}")
            return None
            
    def generate_fallback_analysis(self, report_data: Dict) -> str:
        """Generate fallback analysis when Grok is not available"""
        analysis = f"""
# Job Market Topic Analysis Report

## Executive Summary
Analysis of {report_data['metadata']['total_documents']} job postings identified {report_data['metadata']['n_clusters']} distinct market segments using {report_data['metadata']['algorithm']} clustering.

## Key Findings

### Cluster Overview:
"""
        
        for cluster_id, chars in report_data['cluster_characteristics'].items():
            cluster_size_pct = (chars['size'] / report_data['metadata']['total_documents']) * 100
            analysis += f"""
**Cluster {cluster_id}** ({chars['size']} jobs, {cluster_size_pct:.1f}%):
- Primary Sectors: {', '.join(list(chars['top_sectors'].keys())[:3])}
- Key Locations: {', '.join(list(chars['top_locations'].keys())[:3])}
- Main Source: {max(chars['source_distribution'].keys(), key=lambda x: chars['source_distribution'][x])}
"""

        analysis += """
## Strategic Insights

### Geographic Distribution
- Casablanca dominates the job market across most clusters
- Regional specialization patterns identified
- Remote work opportunities vary by cluster

### Sector Analysis
- Traditional sectors show strong representation
- Technology and service sectors growing
- Cross-sector skill requirements emerging

### Recommendations

**For Job Seekers:**
- Focus on clusters with high demand in your location
- Develop skills aligned with cluster-specific requirements
- Consider geographic mobility for better opportunities

**For Employers:**
- Target recruitment in dominant clusters for your sector
- Consider location-based hiring strategies
- Adapt job descriptions to cluster-specific language

**For Policy Makers:**
- Address geographic imbalances in job distribution
- Support skill development in high-demand areas
- Encourage sector diversification in regions
"""
        
        return analysis
        
    def create_pdf_report(self, report_data: Dict, ai_analysis: str, output_path: str) -> str:
        """Create comprehensive PDF report using ReportLab"""
        try:
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Story container for all content
            story = []
            
            # Title
            story.append(Paragraph("Job Market Topic Analysis Report", self.styles['CustomTitle']))
            story.append(Spacer(1, 12))
            
            # Metadata table
            metadata_data = [
                ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Algorithm:', report_data['metadata']['algorithm']],
                ['Documents:', str(report_data['metadata']['total_documents'])],
                ['Clusters:', str(report_data['metadata']['n_clusters'])],
                ['Features:', str(report_data['metadata']['features_used'])]
            ]
            
            metadata_table = Table(metadata_data, colWidths=[2*inch, 3*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(metadata_table)
            story.append(Spacer(1, 20))
            
            # AI Analysis Section
            story.append(Paragraph("AI-Generated Analysis", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))
            
            # Process AI analysis text with enhanced formatting
            self._process_ai_analysis_text(ai_analysis, story)
            
            # Page break before technical details
            story.append(PageBreak())
            
            # Technical Details Section
            story.append(Paragraph("Technical Analysis Details", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))
            
            # Cluster Details
            for cluster_id, topics in report_data['cluster_topics'].items():
                story.append(Paragraph(f"Cluster {cluster_id} ({topics['size']} jobs)", self.styles['CustomSubheading']))
                
                # Keywords
                keywords_text = f"<b>Keywords:</b> {', '.join(topics['keywords'][:15])}"
                story.append(Paragraph(keywords_text, self.styles['CustomBody']))
                
                # Characteristics
                chars = report_data['cluster_characteristics'][cluster_id]
                sectors_text = f"<b>Top Sectors:</b> {', '.join(list(chars['top_sectors'].keys())[:5])}"
                story.append(Paragraph(sectors_text, self.styles['CustomBody']))
                
                locations_text = f"<b>Top Locations:</b> {', '.join(list(chars['top_locations'].keys())[:5])}"
                story.append(Paragraph(locations_text, self.styles['CustomBody']))
                
                story.append(Spacer(1, 12))
            
            # Summary Insights
            story.append(PageBreak())
            story.append(Paragraph("Key Insights Summary", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))
            
            for insight in report_data['summary_insights']:
                story.append(Paragraph(f"• {insight}", self.styles['BulletPoint']))
            
            # Build PDF
            doc.build(story)
            
            return "PDF report generated successfully"
            
        except Exception as e:
            logger.error(f"Error creating PDF report: {e}")
            raise
        
    def _process_ai_analysis_text(self, ai_analysis: str, story: list):
        """Process AI analysis text with enhanced formatting"""
        lines = ai_analysis.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if not line:
                i += 1
                continue
            
            # Handle numbered sections (1., 2., 3., etc.)
            if line and line[0].isdigit() and '. ' in line:
                # Main section heading
                section_title = line.split('. ', 1)[1] if '. ' in line else line
                story.append(Paragraph(section_title, self.styles['CustomSubheading']))
                story.append(Spacer(1, 8))
                
            # Handle subsections with colons (Key Findings:, etc.)
            elif line.endswith(':') and not line.startswith('-') and not line.startswith('•'):
                story.append(Paragraph(f"<b>{line}</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 4))
                
            # Handle bold text with ** markers
            elif line.startswith('**') and line.endswith('**'):
                bold_text = line.replace('**', '').strip()
                story.append(Paragraph(f"<b>{bold_text}</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 4))
                
            # Handle bullet points starting with -
            elif line.startswith('- '):
                bullet_text = line[2:].strip()
                # Handle bold text within bullet points
                if '**' in bullet_text:
                    bullet_text = bullet_text.replace('**', '<b>', 1).replace('**', '</b>', 1)
                story.append(Paragraph(f"• {bullet_text}", self.styles['BulletPoint']))
                
            # Handle bullet points starting with •
            elif line.startswith('• '):
                bullet_text = line[2:].strip()
                if '**' in bullet_text:
                    bullet_text = bullet_text.replace('**', '<b>', 1).replace('**', '</b>', 1)
                story.append(Paragraph(f"• {bullet_text}", self.styles['BulletPoint']))
                
            # Handle cluster descriptions with **Cluster X:**
            elif line.startswith('**Cluster') and ':' in line:
                cluster_text = line.replace('**', '').strip()
                story.append(Paragraph(f"<b>{cluster_text}</b>", self.styles['CustomBody']))
                story.append(Spacer(1, 4))
                
            # Handle regular paragraphs
            elif line and not line.startswith('#'):
                # Process inline bold formatting
                formatted_line = line
                if '**' in formatted_line:
                    # Replace **text** with <b>text</b>
                    import re
                    formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', formatted_line)
                
                story.append(Paragraph(formatted_line, self.styles['CustomBody']))
                story.append(Spacer(1, 3))
            
            i += 1
        
    def save_report(self, report_data: Dict, output_path: str) -> bool:
        """Generate and save the complete report"""
        try:
            # Generate Grok analysis
            prompt = self.generate_grok_prompt(report_data)
            ai_analysis = self.call_grok_api(prompt)
            
            if not ai_analysis:
                logger.info("Using fallback analysis instead of Grok")
                ai_analysis = self.generate_fallback_analysis(report_data)
            
            logger.info("AI analysis generated successfully")
            ai_analysis = self.clean_ai_analysis_text(ai_analysis)
            
            # Create PDF
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.create_pdf_report(report_data, ai_analysis, output_path)
            
            # Save analysis as markdown for reference
            md_path = output_path.replace('.pdf', '_analysis.md')
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(ai_analysis)
            
            logger.info(f"Report saved to {output_path}")
            logger.info(f"Analysis saved to {md_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return False

def generate_comprehensive_report(report_data: Dict, output_path: str = 'reports/topic_analysis_report.pdf') -> bool:
    """Main function to generate comprehensive topic analysis report"""
    generator = TopicAnalysisReportGenerator()
    return generator.save_report(report_data, output_path)

def main():
    """Test the report generator"""
    # Sample data for testing
    sample_data = {
        'metadata': {
            'algorithm': 'KMeans',
            'n_clusters': 3,
            'total_documents': 250,
            'features_used': 500,
            'analysis_date': datetime.now().isoformat()
        },
        'cluster_topics': {
            0: {'keywords': ['sales', 'commercial', 'client'], 'size': 80},
            1: {'keywords': ['it', 'software', 'developer'], 'size': 90},
            2: {'keywords': ['admin', 'secretary', 'office'], 'size': 80}
        },
        'cluster_characteristics': {
            0: {'size': 80, 'top_sectors': {'Commercial': 40}, 'top_locations': {'Casablanca': 50}, 'source_distribution': {'MarocAnnonce': 60}},
            1: {'size': 90, 'top_sectors': {'IT': 70}, 'top_locations': {'Casablanca': 60}, 'source_distribution': {'Rekrute': 50}},
            2: {'size': 80, 'top_sectors': {'Admin': 60}, 'top_locations': {'Rabat': 40}, 'source_distribution': {'MarocAnnonce': 70}}
        },
        'summary_insights': [
            'IT cluster shows strong growth potential',
            'Geographic concentration in major cities',
            'Source platforms show sector preferences'
        ]
    }
    
    generate_comprehensive_report(sample_data, 'reports/test_report.pdf')
    print("Test report generated!")

if __name__ == "__main__":
    main() 