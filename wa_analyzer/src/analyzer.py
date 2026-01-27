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
        self.data = data.copy()
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
        Refined Logic:
        - True Ghost: I read it (read_at present) but didn't reply > threshold.
        - Left on Delivered: I didn't read it (read_at is null) and didn't reply > threshold.
        """
        df = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
        
        df['next_from_me'] = df['from_me'].shift(-1)
        df['next_jid'] = df['jid_row_id'].shift(-1)
        df['next_timestamp'] = df['timestamp'].shift(-1)
        
        df['time_to_next'] = (df['next_timestamp'] - df['timestamp']).dt.total_seconds()
        
        # Their Messages (From Them)
        their_msgs = df[df['from_me'] == 0]
        
        # Ignored Condition: No Reply OR Delayed Reply
        # Note: Original Ghosting logic considered "Delayed Reply" as ghosting too if > threshold.
        # "Left on Delivered" implies "Unopened".
        ignored = their_msgs[
            (their_msgs['next_jid'] != their_msgs['jid_row_id']) | 
            (their_msgs['time_to_next'].isna()) | 
            (their_msgs['time_to_next'] > threshold_seconds)
        ]
        
        if ignored.empty: return pd.DataFrame()
        
        # Check if 'read_at' exists
        if 'read_at' in self.data.columns:
            ignored = ignored.copy()
            # If read_at is not null -> True Ghost 👻
            # If read_at is null -> Left on Delivered 📨
            ignored.loc[:, 'type'] = ignored['read_at'].apply(lambda x: 'True Ghost 👻' if pd.notnull(x) else 'Left on Delivered 📨')
        else:
            # Fallback if no receipts data
            ignored['type'] = 'Ignored (No Receipts)'
            
        stats = ignored.groupby(['contact_name', 'type']).size().unstack(fill_value=0)
        
        # Add Totals
        stats['Total Ignored'] = stats.sum(axis=1)
        
        # Add Gender
        gender_map = self.data.drop_duplicates('contact_name').set_index('contact_name')['gender']
        stats['gender'] = stats.index.map(gender_map)
        
        return stats.sort_values('Total Ignored', ascending=False)

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
        
        # Optimization: Join all text first, then apply regex once.
        # This is significantly faster than using .apply() on every row.
        full_text = " ".join(text_data.tolist())
        
        # Remove URLs (http/https and www.)
        full_text = re.sub(r'http\S+|www\.\S+', '', full_text)
        
        # Remove Emails if requested
        if exclude_emails:
            full_text = re.sub(r'\S+@\S+', '', full_text)
            
        words = full_text.lower().split()
        # Filter stopwords and short words
        valid_words = [w for w in words if w not in STOPWORDS and len(w) >= min_word_length]
        
        return " ".join(valid_words)

    def get_behavioral_scorecard(self):
        """
        Returns a DataFrame with behavioral metrics per contact:
        - night_owl_pct: % msgs sent 00:00-06:00
        - early_bird_pct: % msgs sent 06:00-09:00
        - double_text_ratio: "Consecutive turns" / "Total turns" ratio
        """
        if self.data.empty: return pd.DataFrame()
        
        df = self.data.copy()
        
        # 1. Time based Metrics
        df['hour'] = df['timestamp'].dt.hour
        total_counts = df['contact_name'].value_counts()
        
        night_counts = df[(df['hour'] >= 0) & (df['hour'] < 6)]['contact_name'].value_counts()
        early_counts = df[(df['hour'] >= 6) & (df['hour'] < 9)]['contact_name'].value_counts()
        
        stats = pd.DataFrame({
            'total': total_counts,
            'night_owl': night_counts,
            'early_bird': early_counts
        }).fillna(0)
        
        stats['night_owl_pct'] = (stats['night_owl'] / stats['total']) * 100
        stats['early_bird_pct'] = (stats['early_bird'] / stats['total']) * 100
        
        # 2. Double Text Ratio (Refined: Turns with >1 message)
        # Old logic ("Resumed Monologue") yielded 99%.
        # New Logic: A "Turn" is a sequence of consecutive messages by same sender.
        # Ratio = (Turns with > 1 message) / (Total Turns)
        
        df_sorted = df.sort_values(['jid_row_id', 'timestamp'])
        
        # Identify changes in sender OR chat (JID)
        # Shift to compare with previous
        df_sorted['prev_sender'] = df_sorted['contact_name'].shift(1)
        df_sorted['prev_jid'] = df_sorted['jid_row_id'].shift(1)
        
        # A new turn starts if Sender changes OR JID changes
        # Strict "Conversation Turn" definition
        change_mask = (df_sorted['contact_name'] != df_sorted['prev_sender']) | (df_sorted['jid_row_id'] != df_sorted['prev_jid'])
        
        # Assign Turn ID
        df_sorted['turn_id'] = change_mask.cumsum()
        
        # Aggregation: Count msgs per turn
        # Group by Turn ID -> (Contact, Count)
        turn_stats = df_sorted.groupby(['turn_id', 'contact_name']).size().reset_index(name='msg_count')
        
        # Now analyze per contact
        # Total Turns
        total_turns = turn_stats['contact_name'].value_counts()
        
        # Double Text Turns (count > 1)
        multi_msg_turns = turn_stats[turn_stats['msg_count'] > 1]['contact_name'].value_counts()
        
        stats['total_turns'] = total_turns
        stats['double_texts'] = multi_msg_turns
        stats = stats.fillna(0)
        
        stats['double_text_ratio'] = 0.0
        mask_turns = stats['total_turns'] > 5
        stats.loc[mask_turns, 'double_text_ratio'] = (stats.loc[mask_turns, 'double_texts'] / stats.loc[mask_turns, 'total_turns']) * 100
        
        # Add Gender
        gender_map = self.data.drop_duplicates('contact_name').set_index('contact_name')['gender']
        stats['gender'] = stats.index.map(gender_map)
        
        return stats

    def get_fun_stats(self, top_n=100):
        """
        Returns stats for:
        - Laughs
        - Emojis (Top used - placeholder logic)
        - Media counts
        - Review/Deleted counts
        - Dry Texter (Avg Length)
        
        top_n: Filter analysis to top N contacts by message volume (default 100)
        """
        df = self.data.copy()
        
        # Filter for Top N Contacts (by volume) to keep stats relevant
        if top_n and top_n > 0:
             top_contacts = df['contact_name'].value_counts().head(top_n).index
             df_top = df[df['contact_name'].isin(top_contacts)].copy()
        else:
             df_top = df
             
        # 1. Laughs
        laugh_pattern = r'(?i)(?:haha|jaja|lol|lmao|risas|xD|😂|🤣)'
        df_top['has_laugh'] = df_top['text_data'].astype(str).str.contains(laugh_pattern, regex=True)
        laugh_counts = df_top[df_top['has_laugh']]['contact_name'].value_counts()
        
        # 2. Media & Deleted
        revoke_pattern = r'(?i)(?:This message was deleted|Eliminaste este mensaje|Se eliminó este mensaje)'
        is_revoked = df_top['text_data'].astype(str).str.fullmatch(revoke_pattern)
        if 'message_type' in df_top.columns:
            is_revoked = is_revoked | (df_top['message_type'] == 15)
            
        deleted_counts = df_top[is_revoked]['contact_name'].value_counts()
        
        # Media
        media_mask = df_top['mime_type'].notnull()
        media_counts = df_top[media_mask]['contact_name'].value_counts()
        
        # 3. Dry Texter (Avg Words)
        # Filter for TEXT messages only (no media, no revokes)
        text_only = df_top[~media_mask & ~is_revoked & df_top['text_data'].notnull()].copy()
        text_only['word_count'] = text_only['text_data'].astype(str).str.split().str.len()
        
        # Filter 0 word length
        text_only = text_only[text_only['word_count'] > 0]
        
        avg_len = text_only.groupby('contact_name')['word_count'].mean()
        avg_len = avg_len[avg_len > 0]
        
        stats = pd.DataFrame({
            'laughs': laugh_counts,
            'media': media_counts,
            'deleted': deleted_counts,
            'avg_word_len': avg_len
        }).fillna(0)
        
        # Add gender
        gender_map = self.data.drop_duplicates('contact_name').set_index('contact_name')['gender']
        stats['gender'] = stats.index.map(gender_map)
        
        return stats

    def get_streak_stats(self):
        """
        Calculates the longest streak of consecutive days with messages.
        Returns Series: contact_name -> longest_streak (int)
        """
        df = self.data.copy()
        df['date'] = df['timestamp'].dt.date
        
        # Get unique dates per contact
        # Group by contact, then find streaks
        # Optimization: Global streak or per contact? 
        # Usually streak is "Me + Them" (Chat Streak). 
        # But here we might want "Who do I have the best streak with?"
        # So group by contact.
        
        results = {}
        
        # Aggregate unique dates per contact
        contact_dates = df.groupby('contact_name')['date'].unique()
        
        for contact, dates in contact_dates.items():
            if len(dates) < 2:
                results[contact] = 1 if len(dates) == 1 else 0
                continue
                
            sorted_dates = sorted(dates)
            
            # Logic for generic streak
            max_streak = 1
            current_streak = 1
            
            for i in range(1, len(sorted_dates)):
                delta = (sorted_dates[i] - sorted_dates[i-1]).days
                if delta == 1:
                    current_streak += 1
                else:
                    max_streak = max(max_streak, current_streak)
                    current_streak = 1
            
            max_streak = max(max_streak, current_streak)
            results[contact] = max_streak
            
        return pd.Series(results, name='longest_streak')

    def get_conversation_killers(self, threshold_seconds=86400):
        """
        Who sent the last message before a long silence?
        """
        df = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
        df['time_to_next'] = df['timestamp'].shift(-1) - df['timestamp']
        df['next_jid'] = df['jid_row_id'].shift(-1)
        
        # Filter: Gap > Threshold OR End of Chat (Next JID diff, or Last msg)
        # Note: "End of Chat" usually means silence until now.
        
        # We consider a "Kill" if silence > 24h.
        # We verify it's the same chat.
        valid_gap = (df['jid_row_id'] == df['next_jid']) & (df['time_to_next'].dt.total_seconds() > threshold_seconds)
        
        # Also include the absolute last message of the chat? 
        # Maybe not "Killer" but "Last Word". Current logic: Silence breakers vs Silence makers.
        
        killers = df[valid_gap]['contact_name'].value_counts()
        return killers

    def get_reaction_stats(self):
        """
        Returns stats about reactions:
        - top_reactors: Who gives most reactions (DataFrame)
        - top_emojis: Most used emojis (Series)
        - most_reacted_msgs: Messages with most reactions (DataFrame)
        """
        if 'reactions_list' not in self.data.columns: return None
        
        # Explode the reactions list
        # reactions_list contains tuples (emoji, sender_raw_string)
        df_reacts = self.data[['message_row_id', 'contact_name', 'reactions_list', 'text_data']].dropna(subset=['reactions_list']).copy()
        
        all_reactions = []
        for idx, row in df_reacts.iterrows():
            for emoji, sender in row['reactions_list']:
                all_reactions.append({
                    'message_id': row['message_row_id'],
                    'chat_contact': row['contact_name'], # The chat it happened in
                    'reaction': emoji,
                    'sender': sender, # RAW JID
                    'preview': str(row['text_data'])[:50]
                })
        
        if not all_reactions: return None
        
        feat_df = pd.DataFrame(all_reactions)
        
        # Top Reactors (by Sender JID) -> Need to map JID to Name if possible, 
        # or just use 'sender' raw string if names aren't perfectly mapped there.
        # Ideally we'd map 'sender' -> 'Display Name' using self.utils or similar map, 
        # but for now we return raw.
        top_reactors = feat_df['sender'].value_counts().head(10)
        
        # Top Emojis
        top_emojis = feat_df['reaction'].value_counts().head(10)
        
        # Most Reacted Messages
        msg_counts = feat_df.groupby(['message_id', 'preview', 'chat_contact']).size().sort_values(ascending=False).head(10).reset_index(name='count')
        
        return {'top_reactors': top_reactors, 'top_emojis': top_emojis, 'most_reacted': msg_counts}

    def get_true_ghosting_stats(self, threshold_hours=24):
        """
        Advanced ghosting:
        - Ghosted: I sent -> They Read it (timestamp) -> No Reply > T hours.
        - Left on Delivered: I sent -> No Read -> No Reply > T hours.
        """
        if 'read_at' not in self.data.columns: return pd.DataFrame()
        
        threshold_seconds = threshold_hours * 3600
        df = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
        
        df['next_from_me'] = df['from_me'].shift(-1)
        df['next_timestamp'] = df['timestamp'].shift(-1)
        df['next_jid'] = df['jid_row_id'].shift(-1)
        
        # Filter for MY messages
        my_msgs = df[df['from_me'] == 1].copy()
        
        # Condition: Next message (reply) doesn't exist OR is too far away
        # AND it's the same chat
        is_ignored = (my_msgs['next_jid'] != my_msgs['jid_row_id']) | \
                     ((my_msgs['next_timestamp'] - my_msgs['timestamp']).dt.total_seconds() > threshold_seconds) | \
                     (my_msgs['next_timestamp'].isna())
                     
        ghost_candidates = my_msgs[is_ignored]
        
        # Classify
        # True Ghost: Read timestamp exists
        # Unseen: Read timestamp is NaT
        ghost_candidates = ghost_candidates.copy()
        ghost_candidates.loc[:, 'type'] = ghost_candidates['read_at'].apply(lambda x: 'True Ghost 👻' if pd.notnull(x) else 'Left on Delivered 📨')
        
        stats = ghost_candidates.groupby(['contact_name', 'type']).size().unstack(fill_value=0)
        if stats.empty: return pd.DataFrame()
        
        stats['Total Ignored'] = stats.sum(axis=1)
        
        # Add Gender
        gender_map = self.data.drop_duplicates('contact_name').set_index('contact_name')['gender']
        stats['gender'] = stats.index.map(gender_map)
        
        return stats.sort_values('Total Ignored', ascending=False)

    def get_advanced_reply_stats(self, limit_hours=8, reply_to=0):
        """
        Returns:
        - Distribution DataFrame (buckets)
        - Fastest/Slowest Responders (Avg Minutes)
        
        reply_to: 
          0 = They are replying to Me (From Them, Prev From Me) -> 'Their Speed'
          1 = I am replying to Them (From Me, Prev From Them) -> 'My Speed'
        """
        df = self.data.sort_values(['jid_row_id', 'timestamp']).copy()
        
        df['prev_from_me'] = df['from_me'].shift(1)
        df['prev_jid'] = df['jid_row_id'].shift(1)
        df['delta'] = (df['timestamp'] - df['timestamp'].shift(1)).dt.total_seconds()
        
        # Determine strict reply condition
        # If reply_to=0 (They -> Me): Current=0, Prev=1
        # If reply_to=1 (Me -> Them): Current=1, Prev=0
        target_from = 0 if reply_to == 0 else 1
        prev_from = 1 if reply_to == 0 else 0
        
        replies = df[
            (df['from_me'] == target_from) & 
            (df['prev_from_me'] == prev_from) & 
            (df['jid_row_id'] == df['prev_jid']) &
            (df['delta'] < limit_hours * 3600) # Cap at limit
        ].copy()
        
        if replies.empty: return None, None
        
        # Buckets (Detailed)
        # <1m, 1-5m, 5-15m, 15m-1h, 1h-4h, 4h-8h, >8h
        bins = [0, 60, 300, 900, 3600, 14400, 28800, max(28801, limit_hours*3600)]
        labels = ['<1m', '1-5m', '5-15m', '15m-1h', '1h-4h', '4h-8h', '>8h']
        
        # Handle small limit cases dynamically
        if limit_hours <= 1:
             bins = [0, 60, 300, 900, 3600]
             labels = ['<1m', '1-5m', '5-15m', '15m-1h']
             
        try:
            replies['bucket'] = pd.cut(replies['delta'], bins=bins, labels=labels, duplicates='drop')
        except ValueError:
            replies['bucket'] = 'Unknown'
            
        distribution = replies.groupby(['contact_name', 'bucket'], observed=False).size().unstack(fill_value=0)
        
        # Averages
        avgs = replies.groupby('contact_name')['delta'].mean() / 60 # Minutes
        
        # Filter sparse contacts
        counts = replies['contact_name'].value_counts()
        valid_contacts = counts[counts >= 5].index
        avgs = avgs[avgs.index.isin(valid_contacts)]
        
        return distribution, avgs
        
    def get_left_on_delivered_stats(self, threshold_seconds=86400):
        """
        Identify who YOU leave on delivered (Unread & Ignored).
        Logic: Last msg from THEM, I never read it (read_at is NaT), and silence > Threshold.
        This implies I ignored it and didn't even open the chat (or receipts off).
        """
        if 'read_at' not in self.data.columns: return pd.DataFrame()
        
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
        
        # Crucial Filter: I did NOT read it.
        # If read_at is valid, then I read it -> "Left on Read" (Ghosting).
        # We want "Left on Delivered" -> read_at is NaT.
        delivered_ignored = ignored[ignored['read_at'].isna()]
        
        counts = delivered_ignored['contact_name'].value_counts().head(20).reset_index()
        counts.columns = ['contact_name', 'count']
        
        # Add gender
        gender_map = self.data.drop_duplicates('contact_name').set_index('contact_name')['gender']
        counts['gender'] = counts['contact_name'].map(gender_map)
        
        return counts

    def get_location_data(self):
        """
        Returns DF with valid locations
        """
        cols = ['latitude', 'longitude', 'contact_name', 'timestamp', 'place_name']
        if 'latitude' not in self.data.columns: return pd.DataFrame()
        
        df = self.data.copy()
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        return df.dropna(subset=['latitude', 'longitude'])[cols]
