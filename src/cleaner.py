"""
Enhanced data cleaning and preprocessing module for ArXiv chatbot project.
Handles comprehensive text cleaning, normalization, and NLP preprocessing.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Set
import logging
from pathlib import Path
import unicodedata
from datetime import datetime
import pytz
import string
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker')

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedArxivDataCleaner:
    """
    Comprehensive data cleaning with advanced NLP preprocessing for ArXiv articles.
    """
    
    def __init__(self, use_spacy: bool = True, language: str = 'en'):
        """
        Initialize the enhanced data cleaner.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP processing
            language: Language for processing (default: 'en')
        """
        self.cleaning_stats = {}
        self.language = language
        self.use_spacy = use_spacy
        
        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Load spaCy model if available
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
                self.nlp = None
        
        # Initialize stop words and common scientific terms
        self.stop_words = self._load_comprehensive_stop_words()
        self.scientific_stop_words = self._load_scientific_stop_words()
        self.latex_patterns = self._compile_latex_patterns()
        self.punctuation_translator = str.maketrans('', '', string.punctuation)
        
        # Academic abbreviations that should be preserved
        self.preserve_abbreviations = {
            'ai', 'ml', 'nlp', 'cv', 'dl', 'nn', 'rnn', 'cnn', 'lstm', 'gru',
            'api', 'gpu', 'cpu', 'ram', 'ssd', 'hdd', 'os', 'ui', 'ux',
            'http', 'https', 'url', 'uri', 'xml', 'json', 'csv', 'pdf',
            'ieee', 'acm', 'arxiv', 'doi', 'issn', 'isbn', 'etc', 'vs',
            'eg', 'ie', 'cf', 'sec', 'fig', 'eq', 'ref', 'def', 'thm',
            'prop', 'lemma', 'cor', 'proof', 'qed', 'iff', 'wrt', 'wlog'
        }
        
    def _load_comprehensive_stop_words(self) -> Set[str]:
        """Load comprehensive stop words including NLTK and custom scientific terms."""
        try:
            nltk_stops = set(stopwords.words('english'))
        except:
            nltk_stops = set()
        
        # Custom stop words for academic/scientific text
        custom_stops = {
            'abstract', 'introduction', 'conclusion', 'results', 'discussion',
            'method', 'methods', 'methodology', 'approach', 'analysis',
            'study', 'research', 'paper', 'article', 'work', 'present',
            'propose', 'show', 'demonstrate', 'prove', 'find', 'obtain',
            'achieve', 'perform', 'conduct', 'investigate', 'examine',
            'consider', 'describe', 'discuss', 'analyze', 'evaluate',
            'compare', 'contrast', 'review', 'survey', 'overview',
            'furthermore', 'moreover', 'however', 'therefore', 'thus',
            'hence', 'consequently', 'nevertheless', 'nonetheless',
            'additionally', 'finally', 'conclusion', 'summary',
            'figure', 'table', 'section', 'chapter', 'appendix',
            'equation', 'formula', 'algorithm', 'procedure', 'step',
            'first', 'second', 'third', 'last', 'next', 'previous',
            'given', 'let', 'assume', 'suppose', 'consider', 'note',
            'observe', 'remark', 'recall', 'remember', 'notice',
            'obviously', 'clearly', 'evidently', 'apparently', 'indeed',
            'actually', 'really', 'truly', 'certainly', 'surely',
            'probably', 'possibly', 'likely', 'perhaps', 'maybe',
            'general', 'specific', 'particular', 'special', 'common',
            'typical', 'usual', 'normal', 'standard', 'basic',
            'simple', 'complex', 'difficult', 'easy', 'hard',
            'good', 'bad', 'better', 'worse', 'best', 'worst',
            'large', 'small', 'big', 'little', 'huge', 'tiny',
            'high', 'low', 'long', 'short', 'wide', 'narrow',
            'many', 'few', 'several', 'various', 'different',
            'similar', 'same', 'other', 'another', 'each',
            'every', 'all', 'some', 'any', 'none', 'no',
            'yes', 'true', 'false', 'correct', 'wrong', 'right'
        }
        
        return nltk_stops.union(custom_stops)
    
    def _load_scientific_stop_words(self) -> Set[str]:
        """Load scientific and mathematical stop words."""
        return {
            'theorem', 'lemma', 'corollary', 'proposition', 'definition',
            'proof', 'qed', 'iff', 'implies', 'therefore', 'hence',
            'let', 'assume', 'suppose', 'given', 'such', 'that',
            'where', 'when', 'which', 'what', 'how', 'why',
            'function', 'variable', 'constant', 'parameter', 'value',
            'number', 'integer', 'real', 'complex', 'matrix', 'vector',
            'set', 'subset', 'element', 'member', 'belong', 'contain',
            'equal', 'equals', 'inequality', 'greater', 'less', 'than',
            'maximum', 'minimum', 'optimal', 'optimum', 'solution',
            'problem', 'constraint', 'objective', 'minimize', 'maximize',
            'subject', 'condition', 'satisfy', 'fulfil', 'meet',
            'requirement', 'criterion', 'criteria', 'standard', 'measure'
        }
    
    def _compile_latex_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for LaTeX removal."""
        patterns = [
            # LaTeX commands with arguments
            re.compile(r'\\[a-zA-Z]+\{[^}]*\}', re.MULTILINE),
            re.compile(r'\\[a-zA-Z]+\[[^\]]*\]\{[^}]*\}', re.MULTILINE),
            # LaTeX environments
            re.compile(r'\\begin\{[^}]*\}.*?\\end\{[^}]*\}', re.DOTALL),
            # Math environments
            re.compile(r'\$\$.*?\$\$', re.DOTALL),
            re.compile(r'\$[^$]*\$', re.MULTILINE),
            # LaTeX commands without arguments
            re.compile(r'\\[a-zA-Z]+\b', re.MULTILINE),
            # Special LaTeX characters
            re.compile(r'\\[^a-zA-Z]', re.MULTILINE),
            # Citations and references
            re.compile(r'\\cite\{[^}]*\}', re.MULTILINE),
            re.compile(r'\\ref\{[^}]*\}', re.MULTILINE),
            re.compile(r'\\label\{[^}]*\}', re.MULTILINE),
            # Equations and formulas
            re.compile(r'\\eq\{[^}]*\}', re.MULTILINE),
            re.compile(r'\\eqref\{[^}]*\}', re.MULTILINE),
        ]
        return patterns
    
    def clean_dataset(self, df: pd.DataFrame, deep_clean: bool = True) -> pd.DataFrame:
        """
        Enhanced cleaning pipeline for the dataset.
        
        Args:
            df: Raw DataFrame to clean
            deep_clean: Whether to perform deep NLP cleaning
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting enhanced data cleaning pipeline...")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Track initial state
        initial_count = len(df_clean)
        self.cleaning_stats['initial_count'] = initial_count
        
        # 1. Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # 2. Enhanced text cleaning
        df_clean = self._enhanced_text_cleaning(df_clean, deep_clean)
        
        # 3. Normalize authors
        df_clean = self._normalize_authors(df_clean)
        
        # 4. Clean categories
        df_clean = self._clean_categories(df_clean)
        
        # 5. Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # 6. Validate dates
        df_clean = self._validate_dates(df_clean)
        
        # 7. Clean DOIs
        df_clean = self._clean_dois(df_clean)
        
        # 8. Filter out invalid entries
        df_clean = self._filter_invalid_entries(df_clean)
        
        # 9. Extract and clean keywords (if deep cleaning enabled)
        if deep_clean:
            df_clean = self._extract_keywords(df_clean)
        
        # 10. Language detection and filtering
        df_clean = self._filter_by_language(df_clean)
        
        # Final statistics
        final_count = len(df_clean)
        self.cleaning_stats['final_count'] = final_count
        self.cleaning_stats['removed_count'] = initial_count - final_count
        self.cleaning_stats['removal_rate'] = (initial_count - final_count) / initial_count
        
        logger.info(f"Enhanced cleaning completed. Removed {initial_count - final_count} articles ({self.cleaning_stats['removal_rate']:.2%})")
        
        return df_clean
    
    def _enhanced_text_cleaning(self, df: pd.DataFrame, deep_clean: bool) -> pd.DataFrame:
        """Enhanced text cleaning with NLP preprocessing."""
        logger.info("Performing enhanced text cleaning...")
        
        # Clean titles
        df['title'] = df['title'].apply(lambda x: self._deep_clean_text(x, deep_clean))
        df['title_clean'] = df['title'].apply(lambda x: self._nlp_preprocess(x, preserve_structure=True))
        
        # Clean summaries
        df['summary'] = df['summary'].apply(lambda x: self._deep_clean_text(x, deep_clean))
        df['summary_clean'] = df['summary'].apply(lambda x: self._nlp_preprocess(x, preserve_structure=False))
        
        # Clean journal references
        df['journal_ref'] = df['journal_ref'].apply(lambda x: self._clean_journal_ref(x))
        
        # Clean comments
        df['comment'] = df['comment'].apply(lambda x: self._clean_text_basic(x))
        
        return df
    
    def _deep_clean_text(self, text: str, deep_clean: bool = True) -> str:
        """
        Deep cleaning of text with comprehensive NLP preprocessing.
        
        Args:
            text: Text to clean
            deep_clean: Whether to perform deep cleaning
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 1. Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # 2. Remove LaTeX commands and math formulas
        for pattern in self.latex_patterns:
            text = pattern.sub(' ', text)
        
        # 3. Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Remove special characters and numbers (optional)
        if deep_clean:
            # Remove standalone numbers but preserve those within words
            text = re.sub(r'\b\d+\b', '', text)
            # Remove most punctuation but preserve sentence structure
            text = re.sub(r'[^\w\s\.\!\?\;\:\,\-\(\)]', ' ', text)
        else:
            # Light cleaning - remove problematic characters
            text = re.sub(r'[^\w\s\.\!\?\;\:\,\-\(\)\[\]\{\}\"\'\/\\]', ' ', text)
        
        # 5. Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # 6. Remove excessive punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text.strip()
    
    def _nlp_preprocess(self, text: str, preserve_structure: bool = True) -> str:
        """
        Advanced NLP preprocessing using NLTK and spaCy.
        
        Args:
            text: Text to preprocess
            preserve_structure: Whether to preserve sentence structure
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text) or not text.strip():
            return ""
        
        # Use spaCy if available for better processing
        if self.use_spacy and self.nlp:
            return self._spacy_preprocess(text, preserve_structure)
        else:
            return self._nltk_preprocess(text, preserve_structure)
    
    def _spacy_preprocess(self, text: str, preserve_structure: bool) -> str:
        """Preprocess text using spaCy."""
        doc = self.nlp(text)
        
        processed_tokens = []
        for token in doc:
            # Skip stop words, punctuation, and spaces
            if (token.is_stop or token.is_punct or token.is_space or 
                token.text.lower() in self.stop_words or
                token.text.lower() in self.scientific_stop_words):
                continue
            
            # Preserve important abbreviations
            if token.text.lower() in self.preserve_abbreviations:
                processed_tokens.append(token.text.lower())
                continue
            
            # Skip very short tokens unless they're important
            if len(token.text) < 2:
                continue
            
            # Use lemmatization for better word normalization
            lemma = token.lemma_.lower()
            
            # Skip if lemma is in stop words
            if lemma in self.stop_words or lemma in self.scientific_stop_words:
                continue
            
            processed_tokens.append(lemma)
        
        if preserve_structure:
            return ' '.join(processed_tokens)
        else:
            return ' '.join(processed_tokens)
    
    def _nltk_preprocess(self, text: str, preserve_structure: bool) -> str:
        """Preprocess text using NLTK."""
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        processed_tokens = []
        for token in tokens:
            # Skip punctuation
            if token in string.punctuation:
                continue
            
            # Skip stop words
            if token in self.stop_words or token in self.scientific_stop_words:
                continue
            
            # Preserve important abbreviations
            if token in self.preserve_abbreviations:
                processed_tokens.append(token)
                continue
            
            # Skip very short tokens
            if len(token) < 2:
                continue
            
            # Lemmatize
            lemma = self.lemmatizer.lemmatize(token)
            
            # Skip if lemma is in stop words
            if lemma in self.stop_words or lemma in self.scientific_stop_words:
                continue
            
            processed_tokens.append(lemma)
        
        return ' '.join(processed_tokens)
    
    def _clean_text_basic(self, text: str) -> str:
        """Basic text cleaning without deep NLP processing."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Remove LaTeX
        for pattern in self.latex_patterns:
            text = pattern.sub(' ', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        return text.strip()
    
    def _clean_journal_ref(self, text: str) -> str:
        """Clean journal reference text."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove LaTeX
        for pattern in self.latex_patterns:
            text = pattern.sub(' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate articles with enhanced similarity detection."""
        initial_count = len(df)
        
        # Remove exact duplicates based on ArXiv ID
        df_clean = df.drop_duplicates(subset=['id'], keep='last')
        
        # Remove title duplicates with fuzzy matching
        df_clean = self._remove_fuzzy_duplicates(df_clean)
        
        removed_count = initial_count - len(df_clean)
        self.cleaning_stats['duplicates_removed'] = removed_count
        
        logger.info(f"Removed {removed_count} duplicate articles")
        return df_clean
    
    def _remove_fuzzy_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove articles with very similar titles using fuzzy matching."""
        from difflib import SequenceMatcher
        
        # Create normalized titles for comparison
        df['title_norm'] = df['title'].str.lower().str.strip()
        df['title_norm'] = df['title_norm'].str.replace(r'[^\w\s]', '', regex=True)
        df['title_norm'] = df['title_norm'].str.replace(r'\s+', ' ', regex=True)
        
        # Remove exact matches
        df_clean = df.drop_duplicates(subset=['title_norm'], keep='first')
        
        # TODO: Add fuzzy matching for near-duplicates (computationally expensive)
        # For now, just remove exact matches
        
        df_clean = df_clean.drop('title_norm', axis=1)
        return df_clean
    
    def _extract_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract keywords from titles and summaries."""
        logger.info("Extracting keywords...")
        
        # Combine title and summary for keyword extraction
        df['combined_text'] = df['title_clean'] + ' ' + df['summary_clean']
        
        # Extract keywords using TF-IDF
        try:
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            
            # Fit on a sample if dataset is too large
            sample_size = min(len(df), 10000)
            sample_texts = df['combined_text'].head(sample_size).fillna('')
            
            tfidf_matrix = vectorizer.fit_transform(sample_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract top keywords for each document
            def extract_doc_keywords(text, n_keywords=10):
                if pd.isna(text) or not text.strip():
                    return ""
                
                try:
                    tfidf_scores = vectorizer.transform([text])
                    scores = tfidf_scores.toarray()[0]
                    
                    # Get top keywords
                    top_indices = scores.argsort()[-n_keywords:][::-1]
                    keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
                    
                    return '; '.join(keywords)
                except:
                    return ""
            
            df['keywords'] = df['combined_text'].apply(extract_doc_keywords)
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            df['keywords'] = ""
        
        return df
    
    def _filter_by_language(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter articles by language (keep English only)."""
        logger.info("Filtering by language...")
        
        initial_count = len(df)
        
        # Simple language detection based on common patterns
        def is_english(text):
            if pd.isna(text) or not text.strip():
                return False
            
            # Check for common English words
            english_indicators = {'the', 'and', 'or', 'of', 'to', 'in', 'a', 'an', 'is', 'are', 'was', 'were'}
            words = set(text.lower().split())
            
            # If at least 20% of first 50 words are English indicators
            first_words = list(words)[:50]
            english_count = sum(1 for word in first_words if word in english_indicators)
            
            return english_count >= len(first_words) * 0.1
        
        # Filter based on title and summary
        df['is_english'] = df['title'].apply(is_english) | df['summary'].apply(is_english)
        df_clean = df[df['is_english']].drop('is_english', axis=1)
        
        filtered_count = initial_count - len(df_clean)
        self.cleaning_stats['language_filtered'] = filtered_count
        
        logger.info(f"Filtered {filtered_count} non-English articles")
        return df_clean
    
    def _normalize_authors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced author normalization."""
        logger.info("Normalizing author names...")
        
        def normalize_author_string(author_string):
            if pd.isna(author_string):
                return ""
            
            # Split authors by various separators
            authors = re.split(r'[;,]|(?:\s+and\s+)', str(author_string))
            
            cleaned_authors = []
            for author in authors:
                # Clean each author name
                author = author.strip()
                
                # Remove common prefixes/suffixes
                author = re.sub(r'^(Dr\.?|Prof\.?|Mr\.?|Ms\.?|Mrs\.?)\s+', '', author, flags=re.IGNORECASE)
                author = re.sub(r'\s+(Jr\.?|Sr\.?|Ph\.?D\.?|M\.?D\.?)$', '', author, flags=re.IGNORECASE)
                
                # Remove extra whitespace
                author = re.sub(r'\s+', ' ', author).strip()
                
                # Skip very short or suspicious names
                if len(author) > 2 and not re.match(r'^[^a-zA-Z]*$', author):
                    cleaned_authors.append(author)
            
            return '; '.join(cleaned_authors)
        
        df['author'] = df['author'].apply(normalize_author_string)
        return df
    
    def _clean_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced category cleaning (robust extraction and fallback)."""
        import ast
        logger.info("Cleaning categories (robust)...")

        # Supprimer la colonne _id si elle existe
        if '_id' in df.columns:
            df = df.drop(columns=['_id'])

        # Nettoyer la colonne 'category' pour obtenir une liste propre
        def extract_first_category(cat):
            if pd.isna(cat):
                return "unknown"
            try:
                # Si c'est une liste sous forme de chaîne, on prend le premier élément
                if isinstance(cat, str) and cat.startswith("["):
                    cat_list = ast.literal_eval(cat)
                    if isinstance(cat_list, list) and len(cat_list) > 0:
                        return cat_list[0]
                # Sinon, on retourne la chaîne telle quelle
                return str(cat).strip()
            except Exception:
                return "unknown"

        # Nettoyer/extraire la catégorie principale
        if 'primary_category' in df.columns:
            df['primary_category'] = df['primary_category'].fillna('').astype(str).str.strip()
            # Si la valeur est vide ou "nan", on tente de la récupérer depuis 'category'
            mask_invalid = (df['primary_category'] == '') | (df['primary_category'].str.lower() == 'nan')
            if 'category' in df.columns:
                df.loc[mask_invalid, 'primary_category'] = df.loc[mask_invalid, 'category'].apply(extract_first_category)
        elif 'category' in df.columns:
            df['primary_category'] = df['category'].apply(extract_first_category)
        else:
            df['primary_category'] = 'unknown'

        # Remplacer les valeurs manquantes ou invalides par 'unknown'
        df['primary_category'] = df['primary_category'].replace(['', 'nan', 'None', None, float('nan')], 'unknown').fillna('unknown')

        # Nettoyer la colonne 'category' (liste à chaîne propre)
        def clean_category_string(cat_string):
            if pd.isna(cat_string):
                return "unknown"
            try:
                if isinstance(cat_string, str) and cat_string.startswith("["):
                    cat_list = ast.literal_eval(cat_string)
                    if isinstance(cat_list, list):
                        return '; '.join([str(c).strip() for c in cat_list if c and str(c).strip().lower() != 'nan'])
                return str(cat_string).strip()
            except Exception:
                return "unknown"
        if 'category' in df.columns:
            df['category'] = df['category'].apply(clean_category_string)

        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced missing value handling."""
        logger.info("Handling missing values...")
        
        # Track missing values
        missing_counts = {}
        for col in df.columns:
            missing_counts[col] = df[col].isna().sum()
        
        self.cleaning_stats['missing_values'] = missing_counts
        
        # Fill missing values
        df['title'] = df['title'].fillna('Untitled')
        df['summary'] = df['summary'].fillna('No summary available')
        df['author'] = df['author'].fillna('Unknown Author')
        df['category'] = df['category'].fillna('unknown')
        df['primary_category'] = df['primary_category'].fillna('unknown')
        df['comment'] = df['comment'].fillna('')
        df['doi'] = df['doi'].fillna('')
        df['journal_ref'] = df['journal_ref'].fillna('')
        
        # Fill cleaned versions
        if 'title_clean' in df.columns:
            df['title_clean'] = df['title_clean'].fillna('')
        if 'summary_clean' in df.columns:
            df['summary_clean'] = df['summary_clean'].fillna('')
        if 'keywords' in df.columns:
            df['keywords'] = df['keywords'].fillna('')
        
        return df
    
    
    def _validate_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced date validation."""
        logger.info("Validating dates...")
        
        # Parse dates if they haven't been parsed yet
        if 'published_date' not in df.columns:
            df['published_date'] = pd.to_datetime(df['published'], errors='coerce')
        if 'updated_date' not in df.columns:
            df['updated_date'] = pd.to_datetime(df['updated'], errors='coerce')
        
        # Count invalid dates
        invalid_published = df['published_date'].isna().sum()
        invalid_updated = df['updated_date'].isna().sum()
        
        self.cleaning_stats['invalid_published_dates'] = invalid_published
        self.cleaning_stats['invalid_updated_dates'] = invalid_updated
        
        # Filter out articles with invalid published dates
        df = df[df['published_date'].notna()]
        
        return df

    def _clean_dois(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced DOI cleaning."""
        logger.info("Cleaning DOIs...")
        
        def clean_doi(doi):
            if pd.isna(doi):
                return None
            
            doi = str(doi).strip()
            
            # Extract DOI from URL if present
            if 'doi.org' in doi:
                doi = doi.split('doi.org/')[-1]
            
            # Remove common prefixes
            doi = re.sub(r'^(doi:?|DOI:?)\s*', '', doi, flags=re.IGNORECASE)
            
            # Basic DOI format validation
            if re.match(r'^10\.\d+/.+', doi):
                return doi
            
            return None
        
        df['doi'] = df['doi'].apply(clean_doi)
        return df
    
    def _filter_invalid_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced filtering of invalid entries."""
        logger.info("Filtering invalid entries...")
        
        initial_count = len(df)
        
        # Remove articles with very short titles
        df = df[df['title'].str.len() >= 3]
        
        # Remove articles with very short summaries
        df = df[df['summary'].str.len() >= 10]
        
        # Remove articles with invalid ArXiv IDs
        df = df[df['id'].str.contains(r'\d+', na=False)]
        
        # Remove articles with future dates
        current_date = datetime.now(pytz.UTC)
        df = df[df['published_date'] <= current_date]
       # Remove articles with suspicious patterns
        df = df[~df['title'].str.contains(r'^[^a-zA-Z]', na=False)]  # No titles with only numbers/symbols
        df = df[~df['summary'].str.contains(r'^[^a-zA-Z]', na=False)]  # No summaries with only numbers/symbols

        # Remove articles with excessive repetition (spam-like)
        def has_excessive_repetition(text, threshold=0.7):
            if pd.isna(text) or len(text) < 20:
                return False
            
            words = text.lower().split()
            if len(words) < 5:
                return False
            
            word_counts = Counter(words)
            most_common_count = word_counts.most_common(1)[0][1]
            repetition_ratio = most_common_count / len(words)
            
            return repetition_ratio > threshold
        
        df = df[~df['title'].apply(has_excessive_repetition)]
        df = df[~df['summary'].apply(has_excessive_repetition)]
        
        filtered_count = initial_count - len(df)
        self.cleaning_stats['filtered_invalid'] = filtered_count
        
        logger.info(f"Filtered out {filtered_count} invalid entries")
        return df
    
    def get_cleaning_stats(self) -> Dict:
        """Get comprehensive statistics about the cleaning process."""
        return self.cleaning_stats
    
    def clean_large_dataset(self, input_path: str, output_path: str, 
                          chunk_size: int = 100000) -> Dict:
        """
        Clean large datasets in chunks to avoid memory issues.
        """
        stats = {}
        cleaned_chunks = []
        
        # Process in chunks
        for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
            logger.info(f"Processing chunk {i+1}...")
            cleaned = self.clean_dataset(chunk)
            cleaned_chunks.append(cleaned)
            
            # Aggregate statistics
            for k, v in self.cleaning_stats.items():
                if isinstance(v, (int, float)):
                    stats[k] = stats.get(k, 0) + v
        
        # Combine cleaned chunks
        df_clean = pd.concat(cleaned_chunks, ignore_index=True)
        
        # Save results
        self.save_cleaned_data(df_clean, output_path)
        
        return stats
    

    @staticmethod
    def _clean_text_wrapper(args):
        """Wrapper function that can be pickled for multiprocessing"""
        text, deep_clean, instance = args
        return instance._deep_clean_text(text, deep_clean)

    def _enhanced_text_cleaning(self, df: pd.DataFrame, deep_clean: bool) -> pd.DataFrame:
        """Optimized text cleaning for large datasets"""
        from multiprocessing import Pool, cpu_count
        
        # Prepare arguments
        title_args = [(text, deep_clean, self) for text in df['title'].fillna('').astype(str)]
        summary_args = [(text, deep_clean, self) for text in df['summary'].fillna('').astype(str)]
        
        with Pool(cpu_count()) as pool:
            df['title'] = pool.map(self._clean_text_wrapper, title_args)
            df['summary'] = pool.map(self._clean_text_wrapper, summary_args)
            
        return df

        def save_cleaned_data(self, df: pd.DataFrame, output_path: str, save_separate_fields: bool = True):
            """
            Save cleaned data with multiple output formats.
            
            Args:
                df: Cleaned DataFrame
                output_path: Path to save the cleaned data
                save_separate_fields: Whether to save separate cleaned fields
            """
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save main dataset
            df.to_csv(output_path, index=False)
            
            # Save separate cleaned fields if requested
            if save_separate_fields:
                # Save only essential columns for NLP model
                essential_cols = ['id', 'title', 'title_clean', 'summary', 'summary_clean', 
                                'author', 'published_date', 'primary_category', 'category']
                
                # Add optional columns if they exist
                optional_cols = ['keywords', 'doi', 'journal_ref']
                for col in optional_cols:
                    if col in df.columns:
                        essential_cols.append(col)
                
                df_essential = df[essential_cols]
                essential_path = output_path.parent / f"essential_{output_path.name}"
                df_essential.to_csv(essential_path, index=False)
                
                # Save text-only version for training
                text_only_path = output_path.parent / f"text_only_{output_path.name}"
                df_text = df[['id', 'title_clean', 'summary_clean', 'primary_category']]
                df_text.to_csv(text_only_path, index=False)
            
            # Save cleaning statistics
            stats_path = output_path.parent / "enhanced_cleaning_stats.json"
            import json
            with open(stats_path, 'w') as f:
                json.dump(self.cleaning_stats, f, indent=2, default=str)
            
            logger.info(f"Saved cleaned data to {output_path}")
            if save_separate_fields:
                logger.info(f"Saved essential fields to {essential_path}")
                logger.info(f"Saved text-only version to {text_only_path}")
            logger.info(f"Saved cleaning statistics to {stats_path}")

    def save_cleaned_data(self, df: pd.DataFrame, output_path: str, save_separate_fields: bool = True):
        """
        Save cleaned data with multiple output formats.
        
        Args:
            df: Cleaned DataFrame
            output_path: Path to save the cleaned data
            save_separate_fields: Whether to save separate cleaned fields
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main dataset
        df.to_csv(output_path, index=False)
        
        # Save separate cleaned fields if requested
        if save_separate_fields:
            # Define base essential columns that should always exist
            essential_cols = ['id', 'title', 'summary', 'author', 'published_date', 'primary_category', 'category']
            
            # Add optional columns if they exist
            optional_cols = ['title_clean', 'summary_clean', 'keywords', 'doi', 'journal_ref']
            for col in optional_cols:
                if col in df.columns:
                    essential_cols.append(col)
            
            df_essential = df[essential_cols]
            essential_path = output_path.parent / f"essential_{output_path.name}"
            df_essential.to_csv(essential_path, index=False)
            
            # Save text-only version for training if clean columns exist
            if 'title_clean' in df.columns and 'summary_clean' in df.columns:
                text_only_path = output_path.parent / f"text_only_{output_path.name}"
                df_text = df[['id', 'title_clean', 'summary_clean', 'primary_category']]
                df_text.to_csv(text_only_path, index=False)
        
        # Save cleaning statistics
        stats_path = output_path.parent / "enhanced_cleaning_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump(self.cleaning_stats, f, indent=2, default=str)
        
        logger.info(f"Saved cleaned data to {output_path}")
        if save_separate_fields:
            logger.info(f"Saved essential fields to {essential_path}")
            if 'title_clean' in df.columns and 'summary_clean' in df.columns:
                logger.info(f"Saved text-only version to {text_only_path}")
        logger.info(f"Saved cleaning statistics to {stats_path}")

class EnhancedTextPreprocessor:
    """
    Advanced text preprocessing specifically for NLP model training.
    """
    
    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True, 
                 min_word_length: int = 2, max_word_length: int = 50):
        """
        Initialize the enhanced text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize words
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Initialize NLP tools
        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        
        # Load stop words
        self.stop_words = self._load_stop_words()
        
        # Compile common patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.number_pattern = re.compile(r'\b\d+\b')
        self.punctuation_pattern = re.compile(r'[^\w\s]')
        
    def _load_stop_words(self) -> Set[str]:
        """Load comprehensive stop words."""
        try:
            nltk_stops = set(stopwords.words('english'))
        except:
            nltk_stops = set()
        
        # Add scientific stop words
        scientific_stops = {
            'paper', 'study', 'research', 'work', 'method', 'approach', 'technique',
            'algorithm', 'model', 'system', 'framework', 'analysis', 'result',
            'experiment', 'evaluation', 'performance', 'comparison', 'application',
            'implementation', 'development', 'design', 'propose', 'present',
            'show', 'demonstrate', 'prove', 'find', 'obtain', 'achieve',
            'perform', 'conduct', 'investigate', 'examine', 'consider',
            'describe', 'discuss', 'analyze', 'evaluate', 'compare'
        }
        
        return nltk_stops.union(scientific_stops)
    
    def preprocess_text(self, text: str, preserve_case: bool = False) -> str:
        """
        Comprehensive text preprocessing for NLP training.
        
        Args:
            text: Text to preprocess
            preserve_case: Whether to preserve original case
            
        Returns:
            Preprocessed text
        """
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # 1. Remove URLs and emails
        text = self.url_pattern.sub('', text)
        text = self.email_pattern.sub('', text)
        
        # 2. Remove numbers (optional)
        text = self.number_pattern.sub('', text)
        
        # 3. Convert to lowercase (unless preserving case)
        if not preserve_case:
            text = text.lower()
        
        # 4. Remove punctuation
        text = self.punctuation_pattern.sub(' ', text)
        
        # 5. Tokenize
        tokens = word_tokenize(text)
        
        # 6. Process tokens
        processed_tokens = []
        for token in tokens:
            # Skip if too short or too long
            if len(token) < self.min_word_length or len(token) > self.max_word_length:
                continue
            
            # Skip stop words
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            
            # Lemmatize
            if self.lemmatize:
                token = self.lemmatizer.lemmatize(token)
            
            # Skip if becomes stop word after lemmatization
            if self.remove_stopwords and token.lower() in self.stop_words:
                continue
            
            processed_tokens.append(token)
        
        return ' '.join(processed_tokens)
    
    def extract_ngrams(self, text: str, n: int = 2) -> List[str]:
        """
        Extract n-grams from text.
        
        Args:
            text: Text to extract n-grams from
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        if pd.isna(text):
            return []
        
        tokens = text.split()
        ngrams = []
        
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    def calculate_text_stats(self, text: str) -> Dict:
        """
        Calculate various text statistics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with text statistics
        """
        if pd.isna(text):
            return {
                'char_count': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0,
                'unique_words': 0,
                'lexical_diversity': 0
            }
        
        text = str(text)
        
        # Basic counts
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        
        # Unique words
        unique_words = len(set(words))
        
        # Lexical diversity (unique words / total words)
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'unique_words': unique_words,
            'lexical_diversity': lexical_diversity
        }


# Utility functions for batch processing
def batch_process_texts(texts: List[str], processor: EnhancedTextPreprocessor, 
                       batch_size: int = 1000) -> List[str]:
    """
    Process texts in batches for memory efficiency.
    
    Args:
        texts: List of texts to process
        processor: Text processor instance
        batch_size: Size of each batch
        
    Returns:
        List of processed texts
    """
    processed_texts = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_processed = [processor.preprocess_text(text) for text in batch]
        processed_texts.extend(batch_processed)
        
        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i + len(batch)} texts")
    
    return processed_texts


def analyze_dataset_quality(df: pd.DataFrame) -> Dict:
    """
    Analyze the quality of the cleaned dataset.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {}
    
    # Basic statistics
    quality_metrics['total_articles'] = len(df)
    quality_metrics['unique_authors'] = df['author'].nunique() if 'author' in df.columns else 0
    quality_metrics['unique_categories'] = df['primary_category'].nunique() if 'primary_category' in df.columns else 0
    
    # Text quality metrics
    if 'title_clean' in df.columns:
        title_stats = df['title_clean'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
        quality_metrics['avg_title_length'] = title_stats.mean()
        quality_metrics['min_title_length'] = title_stats.min()
        quality_metrics['max_title_length'] = title_stats.max()
    
    if 'summary_clean' in df.columns:
        summary_stats = df['summary_clean'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)
        quality_metrics['avg_summary_length'] = summary_stats.mean()
        quality_metrics['min_summary_length'] = summary_stats.min()
        quality_metrics['max_summary_length'] = summary_stats.max()
    
    # Missing value analysis
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            quality_metrics[f'missing_{col}'] = missing_count
    
    # Date range analysis
    if 'published_date' in df.columns:
        dates = pd.to_datetime(df['published_date'], errors='coerce')
        quality_metrics['earliest_date'] = dates.min()
        quality_metrics['latest_date'] = dates.max()
        quality_metrics['date_range_years'] = (dates.max() - dates.min()).days / 365.25
    
    return quality_metrics


if __name__ == "__main__":
    # Example usage
    from data_loader import ArxivDataLoader
    
    # Load sample data
    try:
        loader = ArxivDataLoader("data/raw/articles.csv")
        df = loader.load_data(nrows=5000)  # Load larger sample for testing
        
        print(f"Loaded {len(df)} articles from raw data")
        print(f"Sample columns: {df.columns.tolist()}")
        
        # Enhanced cleaning
        cleaner = EnhancedArxivDataCleaner(use_spacy=True)
        df_clean = cleaner.clean_dataset(df, deep_clean=True)
        
        print(f"After enhanced cleaning: {len(df_clean)} articles")
        
        # Print detailed cleaning statistics
        print("\nEnhanced Cleaning Statistics:")
        stats = cleaner.get_cleaning_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Analyze dataset quality
        print("\nDataset Quality Analysis:")
        quality_metrics = analyze_dataset_quality(df_clean)
        for key, value in quality_metrics.items():
            print(f"  {key}: {value}")
        
        # Save cleaned data with multiple formats
        cleaner.save_cleaned_data(df_clean, "data/processed/articles_enhanced_clean.csv", 
                                save_separate_fields=True)
        
        # Test text preprocessing
        print("\nTesting text preprocessing...")
        preprocessor = EnhancedTextPreprocessor(remove_stopwords=True, lemmatize=True)
        
        sample_text = df_clean['title'].iloc[0] if len(df_clean) > 0 else "Sample text"
        processed_text = preprocessor.preprocess_text(sample_text)
        
        print(f"Original: {sample_text}")
        print(f"Processed: {processed_text}")
        
        # Calculate text statistics
        text_stats = preprocessor.calculate_text_stats(sample_text)
        print(f"Text stats: {text_stats}")
        
    except FileNotFoundError:
        print("Raw data file not found. Please ensure 'data/raw/articles.csv' exists.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()