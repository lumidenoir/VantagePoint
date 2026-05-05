import pandas as pd
import spacy
from transformers import pipeline
from newsapi import NewsApiClient
import random
import logging
import os
from datetime import datetime, timedelta

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

class SentimentEngine:
    def __init__(self, api_key=None):
        logger.info("Initializing Intelligence Engine...")
        try:
            # FinBERT is great for financial news sentiment
            self.analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512
            )

            # Load spacy model for aspect extraction and sentence segmentation
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.warning(f"Spacy model not found. Run 'python -m spacy download en_core_web_sm'. Error: {e}")
                self.nlp = None

            # Initialize NewsAPI
            self.api_key = api_key or os.getenv("NEWS_API_KEY")
            self.newsapi = NewsApiClient(api_key=self.api_key) if self.api_key else None
            if not self.newsapi:
                logger.warning("No NewsAPI key found. Running in simulation mode.")

        except Exception as e:
            logger.error(f"Error initializing engine: {e}")
            raise

    def fetch_market_data(self, keyword, limit=20, days_back=30):
        """Fetch news articles using NewsAPI with fallback to mock data."""
        logger.info(f"Fetching news data for: {keyword}")

        if not self.newsapi:
            return self._generate_simulated_news(keyword, limit)

        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            response = self.newsapi.get_everything(
                q=keyword,
                language='en',
                sort_by='relevancy',
                page_size=limit,
                from_param=from_date,
                exclude_domains='pypi.org,github.com,stackoverflow.com,npmjps.com,sourceforge.net,medium.com'
            )

            if response.get('status') == 'ok' and response.get('totalResults', 0) > 0:
                articles = response['articles']
                data = [{
                    'title': art['title'],
                    'text': art['description'] or art['title'],
                    'source': art['source']['name'],
                    'url': art['url'],
                    'publishedAt': art['publishedAt'],
                    'image': art['urlToImage']
                } for art in articles if art['title']]
                return pd.DataFrame(data)
            else:
                logger.warning("NewsAPI returned no results. Using simulation.")
                return self._generate_simulated_news(keyword, limit)

        except Exception as e:
            logger.error(f"NewsAPI error: {e}. Falling back to simulation.")
            return self._generate_simulated_news(keyword, limit)

    def analyze_intelligence(self, df, keyword=None):
        """Analyze sentiment and extract industrial aspects using full context."""
        if df is None or df.empty:
            return pd.DataFrame()

        combined_texts = (df['title'] + ". " + df['text']).fillna("").tolist()

        # Sentiment Analysis
        results = self.analyzer(combined_texts, top_k=None)

        labels, confidences, pos_scores, neg_scores = [], [], [], []

        for res in results:
            scores_dict = {r['label'].lower(): r['score'] for r in res}
            pos_scores.append(scores_dict.get('positive', 0))
            neg_scores.append(scores_dict.get('negative', 0))

            top_res = max(res, key=lambda x: x['score'])
            labels.append(top_res['label'].upper())
            confidences.append(top_res['score'])

        df['sentiment'] = labels
        df['confidence'] = confidences
        df['pos_score'] = pos_scores
        df['neg_score'] = neg_scores

        logger.info(f"Extracting aspects and batch-analyzing sentences for {len(df)} articles...")
        
        doc_data = []
        all_sentences = []
        
        if self.nlp:
            for _, row in df.iterrows():
                full_text = f"{row['title']}. {row['text']}"
                doc = self.nlp(full_text)
                sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
                all_sentences.extend(sentences)
                doc_data.append({'doc': doc, 'sentences': sentences, 'doc_sentiment': row['sentiment']})
            
            unique_sentences = list(set(all_sentences))
            sentence_sentiment_map = {}
            
            if unique_sentences:
                logger.info(f"Performing batch sentiment analysis on {len(unique_sentences)} unique sentences...")
                batch_results = self.analyzer(unique_sentences, truncation=True)
                for sent, res in zip(unique_sentences, batch_results):
                    sentence_sentiment_map[sent] = res['label'].upper()
            
            aspect_data = []
            for data in doc_data:
                aspects = self._extract_aspects_optimized(data['doc'], data['doc_sentiment'], sentence_sentiment_map, keyword)
                aspect_data.append(aspects)
            
            df['aspects'] = aspect_data
        else:
            df['aspects'] = [[] for _ in range(len(df))]
            
        return df

    def _extract_aspects_optimized(self, doc, doc_sentiment, sentence_sentiment_map, keyword=None):
        """Optimized aspect extraction using pre-calculated sentence sentiments."""
        aspects = []

        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text: continue
            
            sent_sentiment = sentence_sentiment_map.get(sent_text, doc_sentiment)

            for chunk in sent.noun_chunks:
                clean_tokens = [t.text for t in chunk if not t.is_stop and not t.is_punct and len(t.text) > 1]
                if not clean_tokens: continue

                clean_text = " ".join(clean_tokens)
                if keyword and keyword.lower() in clean_text.lower() and len(clean_tokens) == 1:
                    continue
                if len(clean_text) < 3:
                    continue

                text_lower = clean_text.lower()
                category = "General Trends"
                if any(x in text_lower for x in ["tech", "ai", "digital", "data", "cyber", "soft", "app", "platform"]):
                    category = "Tech & Innovation"
                elif any(x in text_lower for x in ["price", "cost", "fed", "econ", "tax", "inflation", "market", "revenue"]):
                    category = "Economy & Policy"
                elif any(x in text_lower for x in ["social", "user", "public", "health", "world", "climate", "customer", "people"]):
                    category = "Social & Global"

                aspects.append({
                    'aspect': clean_text.title(),
                    'sentiment': sent_sentiment,
                    'category': category
                })

        unique_aspects = {v['aspect']:v for v in aspects}.values()
        return list(unique_aspects)

    def aggregate_market_insights(self, df):
        """Aggregate insights for dashboard visualizations."""
        if df.empty or 'aspects' not in df.columns:
            return pd.DataFrame()

        all_aspects = [aspect for aspects in df['aspects'] for aspect in aspects]
        if not all_aspects:
            return pd.DataFrame()

        aspect_df = pd.DataFrame(all_aspects)
        stats = aspect_df.groupby(['aspect', 'category', 'sentiment']).size().unstack(fill_value=0)

        for col in ['POSITIVE', 'NEGATIVE']:
            if col not in stats.columns:
                stats[col] = 0

        stats['total'] = stats['POSITIVE'] + stats['NEGATIVE']
        stats['sentiment_score'] = stats.apply(
            lambda x: (x['POSITIVE'] - x['NEGATIVE']) / x['total'] if x['total'] > 0 else 0,
            axis=1
        )
        return stats.sort_values('total', ascending=False).head(20)

    def _generate_simulated_news(self, keyword, limit):
        """Generate high-fidelity simulated news for research demo."""
        scenarios = [
            {"title": f"Market Analysis: {keyword} continues to dominate Q2 projections", "sentiment": "POSITIVE"},
            {"title": f"New breakthroughs in {keyword} technology announced by leading researchers", "sentiment": "POSITIVE"},
            {"title": f"Supply chain issues could impact {keyword} production in coming months", "sentiment": "NEGATIVE"},
            {"title": f"Investors turn cautious on {keyword} as global demand shifts", "sentiment": "NEGATIVE"},
            {"title": f"Sustainable {keyword} initiatives receiving major venture capital funding", "sentiment": "POSITIVE"},
            {"title": f"Regulatory hurdles for {keyword} startups in EU markets", "sentiment": "NEGATIVE"},
            {"title": f"Why {keyword} is the most talked-about trend in the industry this week", "sentiment": "POSITIVE"},
            {"title": f"Consumer report: {keyword} adoption rates hitting record highs", "sentiment": "POSITIVE"},
            {"title": f"Security vulnerability discovered in legacy {keyword} systems", "sentiment": "NEGATIVE"}
        ]
        sources = ["MarketWatch", "Bloomberg", "TechCrunch", "Reuters", "Financial Times", "WSJ", "The Verge"]

        data = []
        for _ in range(limit):
            scen = random.choice(scenarios)
            data.append({
                'title': scen['title'],
                'text': f"Recent reports indicate significant movement regarding {keyword}. {scen['title']}.",
                'source': random.choice(sources),
                'url': "#",
                'publishedAt': (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
                'image': None
            })
        return pd.DataFrame(data)
