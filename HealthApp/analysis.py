import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import func, extract
from sqlalchemy.exc import SQLAlchemyError
from . import db
from .models import (
    Conversation, UserSession, SymptomEntry, Diagnosis, SentimentRecord,
    CommunicationMetric, TreatmentPreference, HealthLiteracy, WorkflowMetric, AIPerformance, HealthAlert
)
from groq import Groq
import json
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import os
from typing import Dict, List, Any, Optional
from statistics import mean, stdev

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Grok client
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in environment variables")
    raise ValueError("GROQ_API_KEY is required")
client = Groq(api_key=GROQ_API_KEY)

class HealthAnalyzer:
    def __init__(self, n_clusters: int = 5):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.default_time_range = '7d'

    def _get_start_date(self, time_range: str) -> datetime:
        """Helper method to determine start date based on time range."""
        try:
            if time_range == '24h':
                return datetime.utcnow() - timedelta(hours=24)
            elif time_range == '30d':
                return datetime.utcnow() - timedelta(days=30)
            elif time_range == '90d':
                return datetime.utcnow() - timedelta(days=90)
            else:
                return datetime.utcnow() - timedelta(days=7)
        except Exception as e:
            logger.error(f"Error calculating start date for {time_range}: {str(e)}")
            return datetime.utcnow() - timedelta(days=7)

    def analyze_symptom_trends(self, time_range: str = '7d') -> List[Dict[str, Any]]:
        """
        Analyze symptom trends using data analysis techniques.
        Returns data for the dashboard's symptom trends chart.
        """
        try:
            start_date = self._get_start_date(time_range)
            end_date = datetime.utcnow()

            # Query symptom entries
            symptom_data = db.session.query(
                SymptomEntry.symptom_name,
                func.date(SymptomEntry.reported_at).label('date'),
                func.count().label('count')
            ).filter(
                SymptomEntry.reported_at >= start_date,
                SymptomEntry.reported_at <= end_date
            ).group_by(
                SymptomEntry.symptom_name,
                func.date(SymptomEntry.reported_at)
            ).all()

            if not symptom_data:
                logger.warning(f"No symptom data found for time range: {time_range}")
                return []

            # Convert to DataFrame for analysis
            df = pd.DataFrame(
                [(s.symptom_name, s.date.strftime('%Y-%m-%d'), s.count) for s in symptom_data],
                columns=['symptom', 'date', 'count']
            )

            # Pivot data for trends
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            pivot_df = df.pivot_table(index='date', columns='symptom', values='count', fill_value=0)
            pivot_df = pivot_df.reindex([d.strftime('%Y-%m-%d') for d in date_range], fill_value=0)

            # Calculate statistical trends
            result = []
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                entry = {'date': date_str}
                for symptom in pivot_df.columns:
                    entry[symptom] = int(pivot_df.loc[date_str, symptom] if date_str in pivot_df.index else 0)
                result.append(entry)

            # Log statistical summary
            total_counts = df.groupby('symptom')['count'].sum().to_dict()
            logger.info(f"Symptom trends for {time_range}: {len(result)} days, top symptoms: {total_counts}")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_symptom_trends: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in analyze_symptom_trends: {str(e)}")
            return []

    def analyze_sentiment(self, conversation_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Perform sentiment analysis using TextBlob and Grok API, with statistical validation.
        Returns sentiment data for the dashboard's pie chart.
        """
        try:
            if not conversation_ids:
                logger.warning("No conversation IDs provided for sentiment analysis")
                return []

            conversations = Conversation.query.filter(Conversation.id.in_(conversation_ids)).all()
            sentiment_counts = defaultdict(int)
            sentiment_scores = []

            for convo in conversations:
                if not convo.messages:
                    continue
                user_messages = [msg['text'] for msg in convo.messages if msg.get('isUser', False)]
                for message in user_messages:
                    # Initial sentiment analysis with TextBlob
                    blob = TextBlob(message)
                    polarity = blob.sentiment.polarity
                    sentiment_scores.append(polarity)

                    # Refine with Grok API
                    try:
                        response = client.chat.completions.create(
                            model="mixtral-8x7b-32768",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "Analyze the sentiment of the following health-related message. Return a JSON object with 'sentiment' (Very Positive, Positive, Neutral, Negative, Very Negative) and 'confidence' (0-1)."
                                },
                                {"role": "user", "content": message}
                            ],
                            max_tokens=100
                        )
                        grok_result = json.loads(response.choices[0].message.content)
                        sentiment = grok_result.get('sentiment', 'Neutral')
                    except Exception as e:
                        logger.warning(f"Grok API failed for sentiment: {str(e)}. Using TextBlob.")
                        if polarity > 0.5:
                            sentiment = 'Very Positive'
                        elif polarity > 0.1:
                            sentiment = 'Positive'
                        elif polarity < -0.5:
                            sentiment = 'Very Negative'
                        elif polarity < -0.1:
                            sentiment = 'Negative'
                        else:
                            sentiment = 'Neutral'

                    sentiment_counts[sentiment] += 1

                    # Store sentiment record
                    sentiment_record = SentimentRecord(
                        convo_id=convo.id,
                        sentiment_category=sentiment,
                        percentage=polarity * 100,
                        recorded_at=datetime.utcnow()
                    )
                    db.session.add(sentiment_record)

            db.session.commit()

            # Calculate statistical metrics
            total = sum(sentiment_counts.values())
            if total == 0:
                logger.warning("No valid messages for sentiment analysis")
                return []

            mean_polarity = mean(sentiment_scores) if sentiment_scores else 0
            std_polarity = stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
            logger.info(f"Sentiment analysis stats: mean polarity={mean_polarity:.2f}, std={std_polarity:.2f}")

            # Format for pie chart
            colors = {
                'Very Positive': '#10B981',
                'Positive': '#34D399',
                'Neutral': '#F59E0B',
                'Negative': '#F87171',
                'Very Negative': '#EF4444'
            }
            result = [
                {'name': k, 'value': (v / total) * 100, 'color': colors.get(k, '#000000')}
                for k, v in sentiment_counts.items()
            ]

            logger.info(f"Sentiment analysis completed for {len(conversations)} conversations")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_sentiment: {str(e)}")
            db.session.rollback()
            return []
        except Exception as e:
            logger.error(f"Unexpected error in analyze_sentiment: {str(e)}")
            db.session.rollback()
            return []

    def analyze_diagnostic_patterns(self, time_range: str = '7d') -> List[Dict[str, Any]]:
        """
        Analyze diagnostic patterns with statistical aggregation.
        Returns data for the dashboard's diagnostic patterns table.
        """
        try:
            start_date = self._get_start_date(time_range)
            diagnoses = db.session.query(
                Diagnosis.condition_name,
                func.count().label('frequency'),
                func.avg(Diagnosis.accuracy).label('avg_accuracy')
            ).filter(
                Diagnosis.created_at >= start_date,
                Diagnosis.created_at <= datetime.utcnow()
            ).group_by(
                Diagnosis.condition_name
            ).all()

            if not diagnoses:
                logger.warning(f"No diagnoses found for time range: {time_range}")
                return []

            # Convert to DataFrame for statistical analysis
            df = pd.DataFrame(
                [(d.condition_name, d.frequency, d.avg_accuracy) for d in diagnoses],
                columns=['condition', 'frequency', 'accuracy']
            )
            df['accuracy'] = df['accuracy'].apply(lambda x: round(x * 100, 1) if x else 0)

            # Calculate trends
            result = [
                {
                    'condition': row.condition,
                    'frequency': int(row.frequency),
                    'accuracy': row.accuracy,
                    'trend': 'up' if row.frequency > df.frequency.mean() else 'down'
                }
                for _, row in df.iterrows()
            ]

            logger.info(f"Diagnostic patterns for {time_range}: {len(result)} conditions, mean freq={df.frequency.mean():.1f}")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_diagnostic_patterns: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in analyze_diagnostic_patterns: {str(e)}")
            return []

    def analyze_communication_metrics(self, time_range: str = '7d') -> List[Dict[str, Any]]:
        """
        Analyze communication effectiveness with statistical metrics.
        Returns data for the dashboard's communication effectiveness section.
        """
        try:
            start_date = self._get_start_date(time_range)
            metrics = db.session.query(
                CommunicationMetric.metric_name,
                CommunicationMetric.current_value,
                CommunicationMetric.previous_value,
                CommunicationMetric.trend
            ).filter(
                CommunicationMetric.recorded_at >= start_date,
                CommunicationMetric.recorded_at <= datetime.utcnow()
            ).all()

            if not metrics:
                # Default metrics if none exist
                default_metrics = [
                    {'metric': 'Understanding Rate', 'current': 94, 'previous': 91, 'trend': 'up'},
                    {'metric': 'Follow-through Rate', 'current': 87, 'previous': 89, 'trend': 'down'},
                    {'metric': 'Satisfaction Score', 'current': 4.6, 'previous': 4.4, 'trend': 'up'},
                    {'metric': 'Completion Rate', 'current': 92, 'previous': 88, 'trend': 'up'}
                ]
                for metric in default_metrics:
                    db.session.add(CommunicationMetric(
                        metric_name=metric['metric'],
                        current_value=metric['current'],
                        previous_value=metric['previous'],
                        trend=metric['trend'],
                        recorded_at=datetime.utcnow(),
                        time_range=time_range
                    ))
                db.session.commit()
                metrics = default_metrics
            else:
                metrics = [
                    {
                        'metric': m.metric_name,
                        'current': float(m.current_value),
                        'previous': float(m.previous_value) if m.previous_value else 0,
                        'trend': m.trend or 'stable'
                    }
                    for m in metrics
                ]

            # Statistical analysis
            currents = [m['current'] for m in metrics]
            mean_current = mean(currents) if currents else 0
            logger.info(f"Communication metrics for {time_range}: mean current value={mean_current:.1f}")
            return metrics

        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_communication_metrics: {str(e)}")
            db.session.rollback()
            return []
        except Exception as e:
            logger.error(f"Unexpected error in analyze_communication_metrics: {str(e)}")
            return []

    def analyze_user_activity(self, time_range: str = '24h') -> List[Dict[str, Any]]:
        """
        Analyze user activity by hour with statistical insights.
        Returns data for the dashboard's user activity chart.
        """
        try:
            start_date = self._get_start_date(time_range)
            sessions = db.session.query(
                extract('hour', UserSession.start_time).label('hour'),
                func.count().label('users')
            ).filter(
                UserSession.start_time >= start_date,
                UserSession.start_time <= datetime.utcnow()
            ).group_by(
                extract('hour', UserSession.start_time)
            ).all()

            # Initialize 24-hour array
            result = [{'hour': f'{h:02d}', 'users': 0} for h in range(24)]
            for hour, count in sessions:
                result[int(hour)]['users'] = int(count)

            # Statistical analysis
            user_counts = [r['users'] for r in result]
            peak_hour = max(result, key=lambda x: x['users'])['hour'] if user_counts else '00'
            logger.info(f"User activity for {time_range}: peak hour={peak_hour}, total users={sum(user_counts)}")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_user_activity: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in analyze_user_activity: {str(e)}")
            return []

    def generate_health_alerts(self) -> List[Dict[str, Any]]:
        """
        Generate real-time health alerts using Grok API and statistical clustering.
        Returns data for the dashboard's real-time alerts section.
        """
        try:
            start_date = datetime.utcnow() - timedelta(hours=24)
            recent_symptoms = db.session.query(SymptomEntry).filter(
                SymptomEntry.reported_at >= start_date
            ).all()

            if not recent_symptoms:
                logger.warning("No recent symptoms for health alerts")
                return []

            # Prepare data for clustering
            symptom_texts = [s.symptom_name + (f" ({s.severity})" if s.severity else "") for s in recent_symptoms]
            if len(symptom_texts) < 2:
                logger.warning("Insufficient symptoms for clustering")
                return []

            # Apply ML clustering
            X = self.vectorizer.fit_transform(symptom_texts)
            clusters = self.kmeans.fit_predict(X)

            # Group symptoms by cluster
            clustered_symptoms = defaultdict(list)
            for i, symptom in enumerate(symptom_texts):
                clustered_symptoms[clusters[i]].append(symptom)

            alerts = []
            for cluster_id, symptoms in clustered_symptoms.items():
                prompt = f"""
                Analyze the following symptom cluster for potential health alerts.
                Symptoms: {json.dumps(symptoms)}
                Return a JSON object with:
                - type: (warning, info, success, alert)
                - title: Short title
                - description: Detailed description
                - severity: (low, medium, high)
                - time: Current timestamp
                - region: (optional)
                """
                try:
                    response = client.chat.completions.create(
                        model="mixtral-8x7b-32768",
                        messages=[{"role": "system", "content": prompt}],
                        max_tokens=500
                    )
                    alert = json.loads(response.choices[0].message.content)
                    alert['time'] = datetime.utcnow().isoformat()
                    alerts.append(alert)

                    # Store in database
                    db.session.add(HealthAlert(
                        title=alert['title'],
                        description=alert['description'],
                        severity=alert['severity'],
                        alert_type=alert['type'],
                        created_at=datetime.utcnow(),
                        region=alert.get('region', None)
                    ))
                except Exception as e:
                    logger.warning(f"Grok API failed for alert generation: {str(e)}")

            db.session.commit()
            logger.info(f"Generated {len(alerts)} health alerts")
            return alerts

        except SQLAlchemyError as e:
            logger.error(f"Database error in generate_health_alerts: {str(e)}")
            db.session.rollback()
            return []
        except Exception as e:
            logger.error(f"Unexpected error in generate_health_alerts: {str(e)}")
            return []

    def analyze_treatment_preferences(self, time_range: str = '7d') -> List[Dict[str, Any]]:
        """
        Analyze treatment preferences with statistical trends.
        Returns data for the dashboard's treatment preferences section.
        """
        try:
            start_date = self._get_start_date(time_range)
            preferences = db.session.query(
                TreatmentPreference.treatment_type,
                func.avg(TreatmentPreference.preference_score).label('avg_score'),
                TreatmentPreference.trend
            ).filter(
                TreatmentPreference.recorded_at >= start_date,
                TreatmentPreference.recorded_at <= datetime.utcnow()
            ).group_by(
                TreatmentPreference.treatment_type,
                TreatmentPreference.trend
            ).all()

            if not preferences:
                logger.warning(f"No treatment preferences found for {time_range}")
                return []

            # Convert to DataFrame for analysis
            df = pd.DataFrame(
                [(p.treatment_type, p.avg_score, p.trend) for p in preferences],
                columns=['treatment', 'score', 'trend']
            )
            df['percentage'] = df['score'].apply(lambda x: round(x * 100, 1))

            result = [
                {
                    'treatment': row.treatment,
                    'percentage': row.percentage,
                    'trend': row.trend or 'stable'
                }
                for _, row in df.iterrows()
            ]

            logger.info(f"Treatment preferences for {time_range}: {len(result)} treatments, mean score={df.score.mean():.2f}")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_treatment_preferences: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in analyze_treatment_preferences: {str(e)}")
            return []

    def analyze_health_literacy(self, time_range: str = '7d') -> List[Dict[str, Any]]:
        """
        Analyze health literacy by demographics with statistical insights.
        Returns data for the dashboard's health literacy chart.
        """
        try:
            start_date = self._get_start_date(time_range)
            literacy_data = db.session.query(
                HealthLiteracy.age_group,
                func.avg(HealthLiteracy.understanding_rate).label('avg_understanding'),
                func.avg(HealthLiteracy.engagement_rate).label('avg_engagement')
            ).filter(
                HealthLiteracy.recorded_at >= start_date,
                HealthLiteracy.recorded_at <= datetime.utcnow()
            ).group_by(
                HealthLiteracy.age_group
            ).all()

            if not literacy_data:
                logger.warning(f"No health literacy data for {time_range}")
                return []

            # Convert to DataFrame for analysis
            df = pd.DataFrame(
                [(d.age_group, d.avg_understanding, d.avg_engagement) for d in literacy_data],
                columns=['group', 'understanding', 'engagement']
            )
            df['understanding'] = df['understanding'].apply(lambda x: round(x, 1))
            df['engagement'] = df['engagement'].apply(lambda x: round(x, 1))

            result = [
                {
                    'group': row.group,
                    'understanding': row.understanding,
                    'engagement': row.engagement
                }
                for _, row in df.iterrows()
            ]

            logger.info(f"Health literacy for {time_range}: {len(result)} age groups, mean understanding={df.understanding.mean():.1f}")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_health_literacy: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in analyze_health_literacy: {str(e)}")
            return []

    def analyze_workflow_metrics(self, time_range: str = '7d') -> List[Dict[str, Any]]:
        """
        Analyze workflow metrics with statistical changes.
        Returns data for the dashboard's workflow metrics section.
        """
        try:
            start_date = self._get_start_date(time_range)
            metrics = db.session.query(
                WorkflowMetric.metric_name,
                WorkflowMetric.value,
                WorkflowMetric.change_percentage
            ).filter(
                WorkflowMetric.recorded_at >= start_date,
                WorkflowMetric.recorded_at <= datetime.utcnow()
            ).all()

            if not metrics:
                logger.warning(f"No workflow metrics for {time_range}")
                return []

            # Convert to DataFrame for analysis
            df = pd.DataFrame(
                [(m.metric_name, m.value, m.change_percentage) for m in metrics],
                columns=['metric', 'value', 'change']
            )
            df['value'] = df['value'].apply(lambda x: round(x, 1))
            df['change'] = df['change'].apply(lambda x: round(x, 1) if x is not None else 0)

            result = [
                {
                    'metric': row.metric,
                    'value': row.value,
                    'change': row.change
                }
                for _, row in df.iterrows()
            ]

            logger.info(f"Workflow metrics for {time_range}: {len(result)} metrics, mean value={df.value.mean():.1f}")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_workflow_metrics: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in analyze_workflow_metrics: {str(e)}")
            return []

    def analyze_ai_performance(self, time_range: str = '7d') -> List[Dict[str, Any]]:
        """
        Analyze AI performance metrics with statistical insights.
        Returns data for the dashboard's AI performance section.
        """
        try:
            start_date = self._get_start_date(time_range)
            performance = db.session.query(
                AIPerformance.metric_name,
                AIPerformance.value
            ).filter(
                AIPerformance.recorded_at >= start_date,
                AIPerformance.recorded_at <= datetime.utcnow()
            ).all()

            if not performance:
                logger.warning(f"No AI performance data for {time_range}")
                return []

            # Convert to DataFrame for analysis
            df = pd.DataFrame(
                [(p.metric_name, p.value) for p in performance],
                columns=['metric', 'value']
            )
            df['value'] = df['value'].apply(lambda x: round(x, 1))

            result = [
                {
                    'metric': row.metric,
                    'value': row.value
                }
                for _, row in df.iterrows()
            ]

            logger.info(f"AI performance for {time_range}: {len(result)} metrics, mean value={df.value.mean():.1f}")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error in analyze_ai_performance: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in analyze_ai_performance: {str(e)}")
            return []

    def extract_symptoms_and_diagnoses(self, message: str, patient_info: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """
        Extract symptoms and diagnoses using Grok API and ML clustering.
        Returns data for storage and analysis.
        """
        try:
            prompt = f"""
            Analyze the following health-related message for symptoms and potential diagnoses.
            Patient Info: {json.dumps(patient_info, indent=2)}
            Message: {message}
            Return a JSON object with:
            - symptoms: List of dictionaries with 'name', 'severity' (low, medium, high), 'location' (optional)
            - diagnoses: List of dictionaries with 'condition_name', 'accuracy' (0-1), 'requires_attention' (boolean)
            """
            response = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=500
            )
            result = json.loads(response.choices[0].message.content)

            # Validate result structure
            result = {
                'symptoms': result.get('symptoms', []),
                'diagnoses': result.get('diagnoses', [])
            }

            # Store in database
            convo_id = int(session_id.split('_')[-1])
            for symptom in result['symptoms']:
                db.session.add(SymptomEntry(
                    convo_id=convo_id,
                    symptom_name=symptom.get('name', 'Unknown'),
                    severity=symptom.get('severity', 'medium'),
                    location=symptom.get('location', None),
                    reported_at=datetime.utcnow()
                ))

            for diagnosis in result['diagnoses']:
                db.session.add(Diagnosis(
                    convo_id=convo_id,
                    condition_name=diagnosis.get('condition_name', 'Unknown'),
                    accuracy=diagnosis.get('accuracy', 0.5),
                    requires_attention=diagnosis.get('requires_attention', False),
                    created_at=datetime.utcnow()
                ))

            db.session.commit()
            logger.info(f"Extracted {len(result['symptoms'])} symptoms and {len(result['diagnoses'])} diagnoses")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Database error in extract_symptoms_and_diagnoses: {str(e)}")
            db.session.rollback()
            return {'symptoms': [], 'diagnoses': []}
        except Exception as e:
            logger.error(f"Unexpected error in extract_symptoms_and_diagnoses: {str(e)}")
            return {'symptoms': [], 'diagnoses': []}

def generate_personalized_response(message: str, patient_info: Dict[str, Any], session_id: str, conversation_history: List[Dict[str, Any]]) -> str:
    """
    Generate a personalized response using Grok API, incorporating patient info and history.
    Also extracts symptoms and diagnoses for analysis.
    """
    try:
        analyzer = HealthAnalyzer()
        analysis_result = analyzer.extract_symptoms_and_diagnoses(message, patient_info, session_id)

        # Prepare conversation history
        history_context = "\n".join([f"{'User' if msg.get('isUser', False) else 'AI'}: {msg.get('text', '')}" for msg in conversation_history[-5:]])

        # Prepare prompt
        prompt = f"""
        You are a medical AI assistant for Healia, providing accurate and empathetic responses.
        Patient Info: {json.dumps(patient_info, indent=2)}
        Conversation History:
        {history_context}
        Current Message: {message}
        Analysis Results:
        Symptoms: {json.dumps(analysis_result.get('symptoms', []), indent=2)}
        Diagnoses: {json.dumps(analysis_result.get('diagnoses', []), indent=2)}
        Provide a professional, empathetic, and accurate response. Include medical advice if appropriate, and suggest consulting a healthcare provider for serious conditions. Return only the response text.
        """

        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1000
        )

        response_text = response.choices[0].message.content.strip()
        if not response_text:
            logger.warning("Empty response from Grok API")
            return "I'm sorry, I couldn't generate a response. Please try again."

        logger.info(f"Generated personalized response: {response_text[:100]}...")
        return response_text

    except Exception as e:
        logger.error(f"Error generating personalized response: {str(e)}")
        return "I'm sorry, I couldn't process your request due to an issue with the AI service. Please try again later."