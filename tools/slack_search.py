from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from typing import List, Dict
import numpy as np
import faiss
import re

class SlackSearch:
    def __init__(self, token: str):
        if not token or token == "None":
            print("Warning: No Slack token provided. Slack search will be disabled.")
            self.client = None
        else:
            self.client = WebClient(token=token)

    def intelligent_query_transform(self, user_query: str) -> str:
        """Transform natural language query into optimized Slack search syntax using OpenAI"""
        transform_prompt = f"""
        Convert this natural language question into an optimized Slack keyword search query.
        
        Available Slack search operators:
        - from:@username (search messages from specific user)
        - in:#channel (search in specific channel)
        - after:YYYY-MM-DD (messages after date)
        - before:YYYY-MM-DD (messages before date)
        - during:YYYY-MM-DD (messages on specific date)
        - has:link, has:file (messages with attachments)
        
        User question: "{user_query}"
        
        Transform this into an effective Slack keyword search query. If the question mentions:
        - A person's name → use from:@username
        - A channel name → use in:#channel  
        - Time references → use appropriate date operators
        - File/link mentions → use has: operators

        Return only the optimized search query, no explanation.
        """
        try:
            from openai import OpenAI
            import os
            openai_key = os.environ.get("OPENAI_API_KEY")
            if not openai_key:
                return user_query
            openai = OpenAI(api_key=openai_key)
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at converting natural language to Slack keyword search syntax."},
                    {"role": "user", "content": transform_prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            transformed_query = response.choices[0].message.content.strip()
            transformed_query = transformed_query.replace('"', '').replace("'", "")
            return transformed_query
        except Exception as e:
            print(f"Query transformation failed, using original: {e}")
            return user_query

    def enhanced_salt_query(self, user_query: str) -> str:
        """Enhanced query processing with multiple strategies and fallback"""
        
        salt_terms = "-in:customer-sla-breach -in:customer-triage -in:marketing -in:sales -in:support-overflow -in:omnis -in:tofu -in:customer-membership-alerts -in:optys-qual-haul -in:sigma-compete -in:closed-lost-notifications -in:vector-alerts -in:notifications-alerts -cypress -github -sentry -squadcast -syften"
        
        # Strategy 1: Try structured extraction for complex queries
        if any(word in user_query.lower() for word in ["who", "when", "where", "in channel", "from", "by", "shared", "posted", "yesterday", "last week", "files"]):
            try:
                # For now, use simple transformation
                transformed_query = self.intelligent_query_transform(user_query)
                if len(transformed_query.split()) > 1:
                    return f"{transformed_query} {salt_terms}"
            except Exception as e:
                print(f"Structured extraction failed: {e}")
        
        # Strategy 2: Simple intelligent transformation
        try:
            transformed_query = self.intelligent_query_transform(user_query)
            return f"{transformed_query} {salt_terms}"
        except Exception as e:
            print(f"Query transformation failed: {e}")
        
        # Strategy 3: Fallback to original with salt
        salt_terms = "-cypress -github -customer-sla-breach -customer-triage -sentry -squadcast -syften"
        return f"{user_query} {salt_terms}"

    def clean_docs_query(self, user_query: str) -> str:
        """Clean query processing for documentation search"""
        # Remove Slack-specific operators that don't make sense for docs
        slack_operators = ['from:', 'in:', 'after:', 'before:', 'during:', 'has:']
        
        # Simple cleaning - remove Slack operators
        cleaned_query = user_query
        for operator in slack_operators:
            # Remove operator and the word following it
            pattern = rf'\b{re.escape(operator)}\S+\b'
            cleaned_query = re.sub(pattern, '', cleaned_query)
        
        # Clean up extra spaces
        cleaned_query = ' '.join(cleaned_query.split())
        
        # If query becomes empty after cleaning, use original
        return cleaned_query if cleaned_query.strip() else user_query

    def search_messages(self, query: str, count: int = 50) -> List[Dict]:
        """Search Slack messages using the Web API with query salting"""
        if self.client is None:
            print("Slack search disabled - no valid token")
            return []
        
        # Apply query salting
        salted_query = self.enhanced_salt_query(query)
        print(f"Slack search query: {salted_query}")
            
        try:
            response = self.client.search_messages(
                query=salted_query,        # add extra filters like  "is:thread before:yesterday" if you like
                highlight=True,
                sort="score",              # "score" ⇒ relevance, "timestamp" ⇒ recency
                sort_dir="desc",           # best match first; you can drop this—desc is the default
                count=count
            )
            
            if response and response.get('ok'):
                messages = response.get('messages', {}).get('matches', [])
                if messages:
                    contexts = self.extract_thread_contexts(messages)
                    return contexts
            return []
            
        except SlackApiError as e:
            if e.response['error'] == 'not_authed':
                print("Slack API error: Invalid or missing authentication token")
            else:
                print(f"Slack API error: {e.response['error']}")
            return []
        except Exception as e:
            print(f"Unexpected error in Slack search: {e}")
            return []

    def get_thread_context(self, channel_id: str, thread_ts: str) -> List[Dict]:
        """Retrieve thread context for better RAG performance"""
        if self.client is None:
            return []
            
        try:
            response = self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                limit=10
            )
            return response.get('messages', [])
        except SlackApiError:
            return []

    def extract_thread_contexts(self, messages: List[Dict]) -> List[Dict]:
        """Process threads as complete units rather than individual messages"""
        thread_contexts = {}
        standalone_messages = []

        for message in messages:
            thread_ts = message.get('thread_ts')
            
            if thread_ts:
                # This is part of a thread
                if thread_ts not in thread_contexts:
                    # Get full thread context once
                    full_thread = self.get_thread_context(
                        message.get('channel', {}).get('id'), 
                        thread_ts
                    )
                    
                    # Use existing combine_message_and_thread method
                    combined_text = self.combine_message_and_thread(message, full_thread)
                    
                    # Create consolidated thread context
                    thread_contexts[thread_ts] = {
                        'text': message.get('text', ''),
                        'user': message.get('user', ''),
                        'channel': message.get('channel', {}).get('name', ''),
                        'timestamp': thread_ts,
                        'thread_context': full_thread,
                        'permalink': message.get('permalink', ''),
                        'combined_text': combined_text,
                        'source_type': 'slack'
                    }
            else:
                # Standalone message - process normally
                standalone_messages.append(message)
        
        # Process standalone messages using existing extract_message_context logic
        standalone_contexts = []
        for msg in standalone_messages:
            context = {
                'text': msg.get('text', ''),
                'user': msg.get('user', ''),
                'channel': msg.get('channel', {}).get('name', ''),
                'timestamp': msg.get('ts', ''),
                'thread_context': [],
                'permalink': msg.get('permalink', ''),
                'combined_text': self.combine_message_and_thread(msg, []),
                'source_type': 'slack'
            }
            standalone_contexts.append(context)
        
        # Combine thread contexts and standalone messages
        return list(thread_contexts.values()) + standalone_contexts

    def combine_message_and_thread(self, message: Dict, thread_context: List[Dict]) -> str:
        """Combine message with thread context using smart chunking strategies"""
        combined_text = f"Channel: {message.get('channel', {}).get('name', '')}\n"
        combined_text += f"User: {message.get('user', '')}\n"
        combined_text += f"Message: {message.get('text', '')}\n"
        
        if thread_context:
            combined_text += "\nThread Context:\n"
            for reply in thread_context[:5]:
                combined_text += f"- {reply.get('user', '')}: {reply.get('text', '')}\n"
        
        return combined_text

    def rank_contexts_by_relevance(self, user_query: str, contexts: List[Dict]) -> List[Dict]:
        """Rank contexts by semantic similarity using OpenAI embeddings and FAISS"""
        if not contexts:
            return contexts
            
        # This would require OpenAI client, so we'll skip ranking for now
        # and just return contexts as-is
        return contexts