#!/usr/bin/env python3
"""
Advanced Topic Modeling for Job Market Analysis
Optimized for MarocAnnonce and Rekrute data
"""
import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JobTopicModeler:
    def __init__(self, db_path='data/job_market.db'):
        """Initialize the advanced topic modeler"""
        self.db_path = db_path
        self.df = None
        self.vectorizer = None
        self.X = None
        self.X_scaled = None
        self.model = None
        self.clusters = None
        self.cluster_labels = None
        self.algorithm = None
        
    def load_data(self, source_filter: Optional[str] = None):
        """Load data with optional source filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT title, company, location, sector, fonction,
                   experience, education_level, contract_type,
                   salary, description, requirements, source
            FROM jobs WHERE is_active = 1
            """
            
            if source_filter and source_filter != "All Sources":
                query += f" AND source = '{source_filter}'"
                
            self.df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(self.df)} records for topic modeling")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
            
    def preprocess_text(self, text_column='description'):
        """Advanced text preprocessing for topic modeling"""
        if self.df is None:
            return False
            
        # Combine description and requirements for richer analysis
        if text_column == 'combined':
            self.df['text_for_analysis'] = (
                self.df['description'].fillna('') + ' ' + 
                self.df['requirements'].fillna('') + ' ' +
                self.df['sector'].fillna('') + ' ' +
                self.df['fonction'].fillna('')
            )
        else:
            self.df['text_for_analysis'] = self.df[text_column].fillna('')
            
        # Clean text
        self.df['text_clean'] = self.df['text_for_analysis'].apply(self._advanced_text_cleaning)
        
        # Remove empty documents
        self.df = self.df[self.df['text_clean'].str.len() > 10]
        
        logger.info(f"Preprocessed {len(self.df)} documents for analysis")
        return True
        
    def _advanced_text_cleaning(self, text):
        """Advanced text cleaning for better topic modeling"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove emails and URLs
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+', '', text)
        
        # Remove numbers but keep important patterns
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\+\-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def create_features(self, max_features=1000, ngram_range=(1, 2)):
        """Create TF-IDF features optimized for job descriptions"""
        try:
            # Custom stop words for job descriptions
            job_stopwords = [
                'job', 'work', 'company', 'position', 'role', 'opportunity',
                'candidate', 'applicant', 'employee', 'team', 'department',
                'required', 'preferred', 'must', 'should', 'will', 'can',
                'years', 'year', 'experience', 'working', 'good', 'excellent'
            ]
            
            # Combine default English stop words with custom ones
            stop_words = list(ENGLISH_STOP_WORDS.union(job_stopwords))
            
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words=stop_words,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.8,
                lowercase=True,
                token_pattern=r'\b[a-zA-Z][a-zA-Z\+\-]*[a-zA-Z]\b'
            )
            
            self.X = self.vectorizer.fit_transform(self.df['text_clean'])
            
            # Scale for clustering
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(self.X.toarray())
            
            logger.info(f"Created {self.X.shape[1]} features from {self.X.shape[0]} documents")
            return True
        except Exception as e:
            logger.error(f"Error creating features: {e}")
            return False
            
    def find_optimal_clusters(self, max_clusters=10):
        """Find optimal number of clusters using multiple metrics"""
        if self.X_scaled is None:
            return None
            
        metrics = {
            'inertia': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'k_values': list(range(2, max_clusters + 1))
        }
        
        for k in metrics['k_values']:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.X_scaled)
            
            metrics['inertia'].append(kmeans.inertia_)
            metrics['silhouette'].append(silhouette_score(self.X_scaled, cluster_labels))
            metrics['calinski_harabasz'].append(calinski_harabasz_score(self.X_scaled, cluster_labels))
            
        return metrics
        
    def fit_kmeans(self, n_clusters=5):
        """Fit K-means clustering with optimal parameters"""
        try:
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            self.clusters = self.model.fit_predict(self.X_scaled)
            self.algorithm = 'KMeans'
            
            # Calculate cluster quality
            silhouette = silhouette_score(self.X_scaled, self.clusters)
            calinski = calinski_harabasz_score(self.X_scaled, self.clusters)
            
            logger.info(f"KMeans completed: {n_clusters} clusters, Silhouette: {silhouette:.3f}, Calinski-Harabasz: {calinski:.1f}")
            return True
        except Exception as e:
            logger.error(f"Error in KMeans clustering: {e}")
            return False
            
    def fit_dbscan(self, eps=0.5, min_samples=5):
        """Fit DBSCAN clustering with optimal parameters"""
        try:
            self.model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            self.clusters = self.model.fit_predict(self.X_scaled)
            self.algorithm = 'DBSCAN'
            
            n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
            n_noise = list(self.clusters).count(-1)
            
            logger.info(f"DBSCAN completed: {n_clusters} clusters, {n_noise} noise points")
            return True
        except Exception as e:
            logger.error(f"Error in DBSCAN clustering: {e}")
            return False
            
    def extract_cluster_topics(self, n_words=15):
        """Extract meaningful topics for each cluster"""
        if self.clusters is None:
            return {}
            
        cluster_topics = {}
        feature_names = self.vectorizer.get_feature_names_out()
        
        for cluster_id in set(self.clusters):
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue
                
            # Get documents in this cluster
            cluster_mask = self.clusters == cluster_id
            cluster_docs = self.X[cluster_mask]
            
            # Calculate mean TF-IDF scores
            mean_scores = cluster_docs.mean(axis=0).A1
            
            # Get top words
            top_indices = mean_scores.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            top_scores = [mean_scores[i] for i in top_indices]
            
            cluster_topics[cluster_id] = {
                'keywords': top_words,
                'scores': top_scores,
                'size': cluster_mask.sum()
            }
            
        return cluster_topics
        
    def analyze_cluster_characteristics(self):
        """Analyze cluster characteristics using categorical features"""
        if self.clusters is None:
            return {}
            
        cluster_characteristics = {}
        
        for cluster_id in set(self.clusters):
            if cluster_id == -1:
                continue
                
            cluster_mask = self.clusters == cluster_id
            cluster_data = self.df[cluster_mask]
            
            characteristics = {
                'size': len(cluster_data),
                'top_sectors': cluster_data['sector'].value_counts().head(3).to_dict(),
                'top_locations': cluster_data['location'].value_counts().head(3).to_dict(),
                'top_companies': cluster_data['company'].value_counts().head(3).to_dict(),
                'contract_types': cluster_data['contract_type'].value_counts().to_dict(),
                'education_levels': cluster_data['education_level'].value_counts().to_dict(),
                'avg_description_length': cluster_data['description'].str.len().mean(),
                'source_distribution': cluster_data['source'].value_counts().to_dict()
            }
            
            cluster_characteristics[cluster_id] = characteristics
            
        return cluster_characteristics
        
    def visualize_clusters(self):
        """Create comprehensive cluster visualizations"""
        if self.clusters is None:
            return None
            
        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        # logger.info(f"PCA fit completed: {X_pca.shape}")
        # Create main scatter plot
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=self.clusters,
            hover_data={'sector': self.df['sector'], 'location': self.df['location'], 'source': self.df['source']},
            title=f'{self.algorithm} Clustering Results (PCA Visualization)',
            labels={
                'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                'color': 'Cluster'
            }
        )
        # logger.info(f"Visualized clusters: {fig}")
        return fig
        
    def create_cluster_dashboard(self):
        """Create comprehensive cluster analysis dashboard"""
        if self.clusters is None:
            return None
            
        cluster_topics = self.extract_cluster_topics()
        cluster_chars = self.analyze_cluster_characteristics()
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            subplot_titles=[
                'Cluster PCA Visualization',
                'Cluster Size Distribution',
                'Top Sectors by Cluster',
                'Location Distribution',
                'Contract Types',
                'Education Levels'
            ],
            vertical_spacing=0.12
        )
        
        # PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        
        for cluster_id in set(self.clusters):
            if cluster_id == -1:
                continue
            mask = self.clusters == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=X_pca[mask, 0],
                    y=X_pca[mask, 1],
                    mode='markers',
                    name=f'Cluster {cluster_id}',
                    showlegend=True
                ),
                row=1, col=1
            )
            
        # Cluster sizes
        cluster_sizes = [cluster_chars[cid]['size'] for cid in cluster_chars.keys()]
        cluster_ids = list(cluster_chars.keys())
        
        fig.add_trace(
            go.Bar(x=cluster_ids, y=cluster_sizes, name='Cluster Size'),
            row=1, col=2
        )
        
        fig.update_layout(
            height=1200,
            title_text=f"{self.algorithm} Topic Modeling Results Dashboard",
            showlegend=True
        )
        
        return fig
        
    def generate_report_data(self):
        """Generate comprehensive data for PDF report"""
        if self.clusters is None:
            return None
            
        cluster_topics = self.extract_cluster_topics()
        cluster_chars = self.analyze_cluster_characteristics()
        
        # Create analysis summary
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        n_noise = list(self.clusters).count(-1) if -1 in self.clusters else 0
        
        report_data = {
            'metadata': {
                'algorithm': self.algorithm,
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'total_documents': len(self.df),
                'features_used': self.X.shape[1],
                'analysis_date': datetime.now().isoformat()
            },
            'cluster_topics': cluster_topics,
            'cluster_characteristics': cluster_chars,
            'summary_insights': self._generate_insights(cluster_topics, cluster_chars)
        }
        
        return report_data
        
    def _generate_insights(self, cluster_topics, cluster_chars):
        """Generate key insights from the clustering analysis"""
        insights = []
        
        # Overall insights
        total_jobs = sum(char['size'] for char in cluster_chars.values())
        largest_cluster = max(cluster_chars.keys(), key=lambda x: cluster_chars[x]['size'])
        
        insights.append(f"Analysis identified {len(cluster_chars)} distinct job market segments")
        insights.append(f"Largest cluster (#{largest_cluster}) contains {cluster_chars[largest_cluster]['size']} jobs ({cluster_chars[largest_cluster]['size']/total_jobs*100:.1f}%)")
        
        # Sector insights
        all_sectors = {}
        for char in cluster_chars.values():
            for sector, count in char['top_sectors'].items():
                all_sectors[sector] = all_sectors.get(sector, 0) + count
                
        top_sector = max(all_sectors.keys(), key=lambda x: all_sectors[x])
        insights.append(f"'{top_sector}' is the most represented sector across clusters")
        
        # Location insights
        all_locations = {}
        for char in cluster_chars.values():
            for location, count in char['top_locations'].items():
                all_locations[location] = all_locations.get(location, 0) + count
                
        top_location = max(all_locations.keys(), key=lambda x: all_locations[x])
        insights.append(f"'{top_location}' dominates job postings across all clusters")
        
        return insights

def main():
    """Test the topic modeler"""
    modeler = JobTopicModeler()
    
    if modeler.load_data():
        if modeler.preprocess_text('combined'):
            if modeler.create_features():
                # Test KMeans
                if modeler.fit_kmeans(n_clusters=5):
                    report_data = modeler.generate_report_data()
                    print("Topic modeling completed successfully!")
                    print(f"Found {report_data['metadata']['n_clusters']} clusters")

if __name__ == "__main__":
    main() 