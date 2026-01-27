import pandas as pd
import re
from .utils import Utils

# Custom Stopwords (Spanish + Common Protocols)
STOPWORDS = {
    # Protocols & Metadata
    "https", "http", "com", "www", "image", "omitted", "video", "audio", "sticker", "gif",
    
    # Spanish Common Words
    "de", "la", "que", "el", "en", "y", "a", "los", "se", "del", "las", "un", "por", "con", "una",
    "su", "para", "es", "al", "lo", "como", "mas", "o", "pero", "sus", "le", "ha", "me", "si", "sin",
    "sobre", "este", "ya", "entre", "cuando", "todo", "esta", "ser", "son", "dos", "tambien", "era", "muy",
    "anos", "hasta", "desde", "esta", "mi", "porque", "que", "solo", "han", "yo", "hay", "vez", "puede",
    "todos", "asi", "nos", "ni", "parte", "tiene", "eso", "uno", "donde", "bien", "tiempo", "mismo",
    "ese", "ahora", "cada", "e", "vida", "otro", "despues", "te", "otros", "aunque", "esa", "eso",
    "hace", "otra", "gobierno", "tan", "durante", "siempre", "dia", "tanto", "ella", "tres", "si",
    "gran", "pais", "segun", "menos", "mundo", "ano", "antes", "estado", "contra", "sino", "forma",
    "caso", "nada", "hacer", "general", "estaba", "poco", "estos", "presidente", "mayor", "ante",
    "unos", "les", "algo", "hacia", "casa", "ellos", "ayer", "hecho", "primera", "mucho", "mientras",
    "ademas", "quien", "momento", "millones", "esto", "espana", "hombre", "estan", "pues", "hoy",
    "lugar", "madrid", "nacional", "trabajo", "otras", "mejor", "nuevo", "decir", "algunos", "entonces",
    "todas", "dias", "debe", "politica", "como", "casi", "toda", "tal", "largo", "quiere", "fueron", "no"
}

class WhatsappAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.utils = Utils()
        if 'gender' not in self.data.columns:
             self.data['gender'] = self.data['contact_name'].apply(self.utils.guess_gender)
        
        # Identify groups
        # Groups have raw_string ending in @g.us OR non-null subject (if parser provided it)
        # We rely on raw_string mostly.
        if 'raw_string' in self.data.columns:
            self.data['is_group'] = self.data['raw_string'].astype(str).str.endswith('@g.us')
        else:
            self.data['is_group'] = False # Fallback

    def get_basic_stats(self):
        stats = {}
        stats['total_messages'] = len(self.data)
        stats['unique_contacts'] = self.data['contact_name'].nunique()
        
        # Sent vs Received
        sent_count = self.data[self.data['from_me'] == 1].shape[0]
        received_count = self.data[self.data['from_me'] == 0].shape[0]
        stats['sent_count'] = sent_count
        stats['received_count'] = received_count
        
        return stats

    def get_top_talkers(self, n=20, metric='messages'):
        """
        metric: 'messages' or 'words'
        """
        if metric == 'words':
            # Calculate word count locally (avoid modifying self.data permanently unless cached)
            # Use pandas vectorized string operations for speed
            # We filter for text messages only first to be safe/fast
            df_text = self.data[self.data['text_data'].notnull()].copy()
            df_text['wc'] = df_text['text_data'].astype(str).str.split().str.len()
            
            counts = df_text.groupby('contact_name')['wc'].sum().sort_values(ascending=False).head(n).reset_index()
            counts.columns = ['contact_name', 'count']
        else:
            # Return dataframe with count, gender, AND is_group
            counts = self.data['contact_name'].value_counts().head(n).reset_index()
            counts.columns = ['contact_name', 'count']
        
        # Mapping for efficiency
        meta_map = self.data.drop_duplicates('contact_name').set_index('contact_name')[['gender', 'is_group']]
        counts = counts.merge(meta_map, on='contact_name', how='left')
        
        return counts

    def get_hourly_activity(self, split_by=None):
        """
        split_by: 'gender', 'group', 'sender', or None
        """
        if split_by == 'gender':
            return self.data.groupby([self.data['timestamp'].dt.hour, 'gender']).size().unstack(fill_value=0)
        elif split_by == 'group':
            df_temp = self.data.copy()
            df_temp['type'] = df_temp['is_group'].map({True: 'Group', False: 'Individual'})
            return df_temp.groupby([df_temp['timestamp'].dt.hour, 'type']).size().unstack(fill_value=0)
        elif split_by == 'sender':
            df_temp = self.data.copy()
            df_temp['sender'] = df_temp['from_me'].map({1: 'Me', 0: 'Them'})
            return df_temp.groupby([df_temp['timestamp'].dt.hour, 'sender']).size().unstack(fill_value=0)
        else:
            return self.data['timestamp'].dt.hour.value_counts().sort_index()
    
    def get_daily_activity(self):
        return self.data['timestamp'].dt.day_name().value_counts()

    def get_monthly_activity(self, split_by=None):
        """
        split_by: 'gender', 'group', 'sender', or None
        """
        if split_by:
            col = 'gender' if split_by == 'gender' else 'is_group'
            
            if split_by == 'group':
                df_temp = self.data.copy()
                df_temp['type'] = df_temp['is_group'].map({True: 'Group', False: 'Individual'})
                col = 'type'
            elif split_by == 'sender':
                df_temp = self.data.copy()
                df_temp['sender'] = df_temp['from_me'].map({1: 'Me', 0: 'Them'})
                col = 'sender'
                return df_temp.set_index('timestamp').groupby(col).resample('ME').size().unstack(level=0).fillna(0)
            
            return self.data.set_index('timestamp').groupby(col).resample('ME').size().unstack(level=0).fillna(0)
        else:
            return self.data.set_index('timestamp').resample('ME').size()

    def get_behavioral_timeline(self, ghost_thresh=86400, init_thresh=21600):
        """
        Returns monthly counts for:
        - Ghosted by Them (I sent last)
        - Ghosted by Me (They sent last)
        - Initiated by Me
        - Initiated by Them
        """
        df = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
        
        # Calculate time gaps
        df['next_timestamp'] = df['timestamp'].shift(-1)
        df['time_to_next'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds()
        
        df['prev_timestamp'] = df['timestamp'].shift(1)
        df['time_from_prev'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()
        
        # Since we might be analyzing a single chat, the shift works fine.
        # If analyzing global, we'd need to group by jid, but for Chat Explorer this is fine.
        
        # Ghosting
        ghosted_by_them = df[
            (df['from_me'] == 1) & 
            ((df['time_to_next'] > ghost_thresh) | df['time_to_next'].isna())
        ]
        ghosted_by_me = df[
            (df['from_me'] == 0) & 
            ((df['time_to_next'] > ghost_thresh) | df['time_to_next'].isna())
        ]
        
        # Initiations (First msg or > thresh gap)
        initiations = df[(df['time_from_prev'] > init_thresh) | df['time_from_prev'].isna()]
        init_me = initiations[initiations['from_me'] == 1]
        init_them = initiations[initiations['from_me'] == 0]
        
        # Resample
        timeline = pd.DataFrame()
        timeline['Ghosted by Them'] = ghosted_by_them.set_index('timestamp').resample('ME').size()
        timeline['Ghosted by Me'] = ghosted_by_me.set_index('timestamp').resample('ME').size()
        timeline['Initiated by Me'] = init_me.set_index('timestamp').resample('ME').size()
        timeline['Initiated by Them'] = init_them.set_index('timestamp').resample('ME').size()
        
        return timeline.fillna(0)

    def get_activity_over_time_by_contact(self, contact_list):
        """
        Returns monthly activity for specific contacts.
        """
        df_filtered = self.data[self.data['contact_name'].isin(contact_list)]
        return df_filtered.set_index('timestamp').groupby('contact_name').resample('ME').size().unstack(level=0).fillna(0)

    def calculate_response_times(self):
        # Sort by chat and time
        df = self.data.sort_values(['jid_row_id', 'timestamp'])
        
        # Calculate time difference
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        
        # We want time taken for ME to reply to THEM
        df['prev_from_me'] = df['from_me'].shift(1)
        df['prev_jid'] = df['jid_row_id'].shift(1)
        
        reply_times = df[
            (df['from_me'] == 1) & 
            (df['prev_from_me'] == 0) & 
            (df['jid_row_id'] == df['prev_jid'])
        ]['time_diff']
        
        # Filter outliers (> 24h)
        avg_reply = reply_times[reply_times < 86400].mean() 
        return avg_reply / 60 if pd.notnull(avg_reply) else 0

    def calculate_chat_reply_times(self):
        """Calculates avg reply time for Me and Them in the current dataset"""
        df = self.data.sort_values(['timestamp']).copy()
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        df['prev_from_me'] = df['from_me'].shift(1)
        
        # My Reply Time: Current is from Me, Prev is from Them
        my_replies = df[(df['from_me'] == 1) & (df['prev_from_me'] == 0)]
        my_avg = my_replies[my_replies['time_diff'] < 86400]['time_diff'].mean()
        
        # Their Reply Time: Current is from Them, Prev is from Me
        their_replies = df[(df['from_me'] == 0) & (df['prev_from_me'] == 1)]
        their_avg = their_replies[their_replies['time_diff'] < 86400]['time_diff'].mean()
        
        return (my_avg / 60 if pd.notnull(my_avg) else 0), (their_avg / 60 if pd.notnull(their_avg) else 0)

    def calculate_gender_stats(self):
        metrics = []
        for g in ['male', 'female']: 
            g_data = self.data[self.data['gender'] == g]
            if g_data.empty:
                continue
            
            stats = {'gender': g}
            stats['count'] = len(g_data)
            
            # Avg WPM
            text_msgs = g_data[g_data['text_data'].notnull()]['text_data'].astype(str)
            if not text_msgs.empty:
                word_counts = text_msgs.apply(lambda x: len(x.split()))
                stats['avg_wpm'] = word_counts.mean()
            else:
                stats['avg_wpm'] = 0
            
            # Reply Time
            # Reusing the existing function on filtered data is imperfect but sufficient for overview
            # Better approach used in separate method, but keeping this simple for now
            df_sorted = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
            df_sorted['time_diff'] = df_sorted['timestamp'].diff().dt.total_seconds()
            df_sorted['prev_from_me'] = df_sorted['from_me'].shift(1)
            df_sorted['prev_jid'] = df_sorted['jid_row_id'].shift(1)
            
            replies = df_sorted[
                (df_sorted['from_me'] == 1) & 
                (df_sorted['prev_from_me'] == 0) & 
                (df_sorted['jid_row_id'] == df_sorted['prev_jid']) &
                (df_sorted['gender'] == g)
            ]
            
            valid_times = replies['time_diff'][replies['time_diff'] < 86400]
            stats['avg_reply_time'] = (valid_times.mean() / 60) if not valid_times.empty else 0

            # Media Percentage
            if 'mime_type' in g_data.columns:
                media_count = g_data['mime_type'].notnull().sum()
                stats['media_pct'] = (media_count / len(g_data)) * 100
            else:
                stats['media_pct'] = 0
            
            metrics.append(stats)
            
        return pd.DataFrame(metrics)

    def analyze_by_gender(self):
        return self.data.groupby('gender').size()

    def get_ghosting_stats(self, threshold_seconds=86400):
        """
        Identify who leaves you on read.
        Threshold: 24h (86400) or 5 days (432000)
        """
        df = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
        
        df['next_from_me'] = df['from_me'].shift(-1)
        df['next_jid'] = df['jid_row_id'].shift(-1)
        df['next_timestamp'] = df['timestamp'].shift(-1)
        
        df['time_to_next'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds()
        
        my_msgs = df[df['from_me'] == 1]
        
        ghosted = my_msgs[
            (my_msgs['next_jid'] != my_msgs['jid_row_id']) | 
            (my_msgs['time_to_next'].isna()) | 
            (my_msgs['time_to_next'] > threshold_seconds)
        ]
        
        counts = ghosted['contact_name'].value_counts().head(20).reset_index()
        counts.columns = ['contact_name', 'count']
        
        # Add gender
        gender_map = self.data.drop_duplicates('contact_name').set_index('contact_name')['gender']
        counts['gender'] = counts['contact_name'].map(gender_map)
        
        return counts

    def get_left_on_read_stats(self, threshold_seconds=86400):
        """
        Identify who YOU leave on read.
        Logic: Last msg from THEM, > threshold silence or no reply.
        """
        df = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
        
        df['next_from_me'] = df['from_me'].shift(-1)
        df['next_jid'] = df['jid_row_id'].shift(-1)
        df['next_timestamp'] = df['timestamp'].shift(-1)
        
        df['time_to_next'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds()
        
        their_msgs = df[df['from_me'] == 0]
        
        ignored = their_msgs[
            (their_msgs['next_jid'] != their_msgs['jid_row_id']) | 
            (their_msgs['time_to_next'].isna()) | 
            (their_msgs['time_to_next'] > threshold_seconds)
        ]
        
        counts = ignored['contact_name'].value_counts().head(20).reset_index()
        counts.columns = ['contact_name', 'count']
        
        # Add gender
        gender_map = self.data.drop_duplicates('contact_name').set_index('contact_name')['gender']
        counts['gender'] = counts['contact_name'].map(gender_map)
        
        return counts

    def get_initiation_stats(self, threshold_seconds=21600):
        """
        Who starts conversations?
        Threshold: 6h (21600) or 48h (172800)
        """
        df = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        df['prev_jid'] = df['jid_row_id'].shift(1)
        
        initiations = df[
            (df['jid_row_id'] != df['prev_jid']) | 
            (df['time_diff'] > threshold_seconds)
        ]
        
        their_initiations = initiations[initiations['from_me'] == 0]['contact_name'].value_counts()
        my_initiations = initiations[initiations['from_me'] == 1]['contact_name'].value_counts()
        
        stats = pd.DataFrame({'Them': their_initiations, 'Me': my_initiations}).fillna(0)
        stats['Total'] = stats['Them'] + stats['Me']
        stats['% Them'] = (stats['Them'] / stats['Total']) * 100
        
        return stats.sort_values('Total', ascending=False).head(20)

    def get_reply_time_ranking(self, min_messages=20, max_delay_seconds=43200):
        """
        Calculates average reply time per contact.
        Returns DataFrame with cols: contact_name, my_avg, their_avg (in minutes)
        """
        df = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
        
        # Calculate deltas
        df['prev_from_me'] = df['from_me'].shift(1)
        df['prev_jid'] = df['jid_row_id'].shift(1)
        df['prev_timestamp'] = df['timestamp'].shift(1)
        
        df['time_delta'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()
        
        # Filter for valid response pairs (Same JID, Different Sender, Delta < Limit)
        # Limit to max_delay_seconds to exclude "new conversations" ghosting
        valid_responses = df[
            (df['jid_row_id'] == df['prev_jid']) &
            (df['from_me'] != df['prev_from_me']) &
            (df['time_delta'] < max_delay_seconds)
        ]
        
        # Group by contact
        stats = valid_responses.groupby(['contact_name', 'from_me'])['time_delta'].mean().reset_index()
        
        # Valid contacts filter (enough messages total)
        msg_counts = self.data['contact_name'].value_counts()
        valid_contacts = msg_counts[msg_counts >= min_messages].index
        
        stats = stats[stats['contact_name'].isin(valid_contacts)]
        
        # Pivot
        stats_pivot = stats.pivot(index='contact_name', columns='from_me', values='time_delta')
        stats_pivot.columns = ['their_avg', 'my_avg'] # 0=Them, 1=Me
        
        # Convert seconds to minutes
        stats_pivot = stats_pivot / 60
        
        # Merge gender for coloring
        gender_map = self.data.drop_duplicates('contact_name').set_index('contact_name')['gender']
        stats_pivot['gender'] = stats_pivot.index.map(gender_map)
        
        return stats_pivot.dropna().reset_index()

    def get_wordcloud_text(self, contact_name=None, filter_from_me=None, min_word_length=0, exclude_emails=False):
        """
        filter_from_me: True (only My words), False (only Their words), None (All)
        """
        if contact_name:
            df = self.data[self.data['contact_name'] == contact_name]
        else:
            df = self.data
            
        if filter_from_me is not None:
            v_from = 1 if filter_from_me else 0
            df = df[df['from_me'] == v_from]
            
        text_data = df['text_data'].dropna().astype(str)
        
        def clean_word(text):
            # Remove URLs (http/https and www.)
            text = re.sub(r'http\S+|www\.\S+', '', text)
            
            # Remove Emails if requested
            if exclude_emails:
                text = re.sub(r'\S+@\S+', '', text)
                
            words = text.lower().split()
            return " ".join([w for w in words if w not in STOPWORDS and len(w) >= min_word_length])
            
        clean_text = text_data.apply(clean_word)
        full_text = " ".join(clean_text.tolist())
        return full_text
