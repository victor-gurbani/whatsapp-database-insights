import sqlite3
import pandas as pd
import vobject
import os

class WhatsappParser:
    def __init__(self, msgstore_path, wa_path, vcf_path):
        self.msgstore_path = msgstore_path
        self.wa_path = wa_path
        self.vcf_path = vcf_path
        self.conn_msg = None
        self.conn_wa = None

    def connect(self):
        if os.path.exists(self.msgstore_path):
            self.conn_msg = sqlite3.connect(self.msgstore_path)
            # Handle encoding errors gracefully
            self.conn_msg.text_factory = lambda b: b.decode(errors="ignore")
        if os.path.exists(self.wa_path):
            self.conn_wa = sqlite3.connect(self.wa_path)
            self.conn_wa.text_factory = lambda b: b.decode(errors="ignore")

    def parse_messages(self):
        if not self.conn_msg:
            print("Message store connection not established.")
            return pd.DataFrame()

        # Query based on v3 schema analysis, joining media table and chat subject
        # Added chat.subject to identify group names
        # CHANGE: message._id as message_row_id to align with other tables
        query = """
        SELECT 
            message._id as message_row_id,
            message.chat_row_id,
            message.from_me,
            message.timestamp,
            message.text_data,
            message.key_id,
            chat.jid_row_id,
            chat.subject,
            message.message_type,
            message.sender_jid_row_id,
            message_media.mime_type
        FROM message
        LEFT JOIN chat ON message.chat_row_id = chat._id
        LEFT JOIN message_media ON message._id = message_media.message_row_id
        """
        try:
            df = pd.read_sql_query(query, self.conn_msg)
            # Convert timestamp to datetime (typically ms in WhatsApp)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"Error parsing messages: {e}")
            return pd.DataFrame()

    def parse_jids(self):
        if not self.conn_msg: return pd.DataFrame()
        try:
            return pd.read_sql_query("SELECT _id as jid_row_id, raw_string, user FROM jid", self.conn_msg)
        except:
             # Fallback for old schema
             try: 
                 return pd.read_sql_query("SELECT _id as jid_row_id, raw_string FROM jid", self.conn_msg)
             except: return pd.DataFrame()

    def parse_wa_contacts(self):
        if not self.conn_wa: return pd.DataFrame()
        query = "SELECT jid, display_name FROM wa_contacts"
        try:
            return pd.read_sql_query(query, self.conn_wa)
        except: return pd.DataFrame()

    def parse_vcf(self):
        contacts = {}
        if os.path.exists(self.vcf_path):
            with open(self.vcf_path, 'r') as f:
                content = f.read()
                # Simple extraction
                # FN:Name
                # TEL;...:Number
                # This is a naive parse, ideally use vobject but keeping it simple/fast
                
                # Split by BEGIN:VCARD
                cards = content.split('BEGIN:VCARD')
                for card in cards:
                    name = None
                    tels = []
                    lines = card.splitlines()
                    for line in lines:
                        if line.startswith('FN:'):
                            name = line[3:]
                        elif line.startswith('TEL'):
                            # Extract number
                            parts = line.split(':')
                            if len(parts) > 1:
                                num = parts[-1].replace(' ', '').replace('-', '')
                                tels.append(num)
                    
                    if name and tels:
                        for t in tels:
                            # Normalize: remove +
                            t_norm = t.replace('+', '')
                            contacts[t_norm] = name
                            # Also store without country code if possible? Hard to guess.
                            # Just store last 9?
                            if len(t_norm) > 9:
                                contacts[t_norm[-9:]] = name
                    pass
        return contacts

    def parse_reactions(self):
        """
        Parses reactions via message_add_on tables.
        Returns DataFrame with: message_row_id, reaction, reaction_sender_jid, reaction_timestamp
        """
        if not self.conn_msg: return pd.DataFrame()
        
        # message_add_on_reaction -> message_add_on -> message
        query = """
        SELECT 
            mao.parent_message_row_id as message_row_id,
            mar.reaction,
            mao.sender_jid_row_id,
            mao.timestamp as reaction_timestamp
        FROM message_add_on_reaction mar
        JOIN message_add_on mao ON mar.message_add_on_row_id = mao._id
        """
        try:
            df = pd.read_sql_query(query, self.conn_msg)
            return df
        except Exception as e:
            print(f"Error parsing reactions: {e}")
            return pd.DataFrame()

    def parse_receipts(self):
        """
        Parses reading receipts.
        Join with message table to link key_id -> message_row_id
        """
        if not self.conn_msg: return pd.DataFrame()
        
        # Schema for receipts: _id, key_remote_jid, key_id, ...
        # Message has key_id too.
        query = """
        SELECT 
            m._id as message_row_id,
            r.read_device_timestamp as read_timestamp,
            r.played_device_timestamp as played_timestamp
        FROM receipts r
        JOIN message m ON r.key_id = m.key_id
        WHERE r.read_device_timestamp > 0 OR r.played_device_timestamp > 0
        """
        try:
            df = pd.read_sql_query(query, self.conn_msg)
            return df
        except Exception as e:
            print(f"Error parsing receipts: {e}")
            return pd.DataFrame()

    def parse_mentions(self):
        if not self.conn_msg: return pd.DataFrame()
        query = """
        SELECT message_row_id, jid_row_id as mentioned_jid_row_id FROM message_mentions
        """
        try:
             return pd.read_sql_query(query, self.conn_msg)
        except: return pd.DataFrame()

    def parse_locations(self):
        if not self.conn_msg: return pd.DataFrame()
        query = """
        SELECT message_row_id, latitude, longitude, place_name, place_address FROM message_location
        """
        try:
             return pd.read_sql_query(query, self.conn_msg)
        except: return pd.DataFrame()

    def parse_jids(self):
        if not self.conn_msg:
            return pd.DataFrame()
            
        query = """
        SELECT _id as jid_row_id, raw_string, user FROM jid
        """
        try:
            df = pd.read_sql_query(query, self.conn_msg)
            return df
        except Exception as e:
            print(f"Error parsing JIDs: {e}")
            return pd.DataFrame()

    def parse_wa_contacts(self):
        """Parses the wa.db for display names associated with JIDs"""
        if not self.conn_wa:
            print("WA DB connection not established.")
            return pd.DataFrame()

        query = """
        SELECT jid, display_name, wa_name FROM wa_contacts
        """
        try:
            df = pd.read_sql_query(query, self.conn_wa)
            return df
        except Exception as e:
            print(f"Error parsing WA contacts: {e}")
            return pd.DataFrame()

    def parse_vcf(self):
        if not self.vcf_path or not os.path.exists(self.vcf_path):
            return {}

        contacts = {}
        with open(self.vcf_path, 'r') as f:
            for vcard in vobject.readComponents(f):
                try:
                    name = vcard.fn.value
                    # Extract phones
                    if hasattr(vcard, 'tel'):
                        for tel in vcard.tel_list:
                            phone_raw = str(tel.value)
                            # Normalize: keep strict digits
                            digits = "".join(filter(str.isdigit, phone_raw))
                            
                            # Store both full digits and last 9 digits (for matching without country code)
                            if len(digits) > 5:
                                contacts[digits] = name
                                if len(digits) > 9:
                                    contacts[digits[-9:]] = name
                except Exception:
                    pass
        return contacts

    def parse_reactions(self):
        """
        Parses reactions via message_add_on tables.
        Returns DataFrame with: message_row_id, reaction, reaction_sender_jid, reaction_timestamp
        """
        if not self.conn_msg: return pd.DataFrame()
        
        # message_add_on_reaction -> message_add_on -> message
        query = """
        SELECT 
            mao.parent_message_row_id as message_row_id,
            mar.reaction,
            mao.sender_jid_row_id,
            mao.timestamp as reaction_timestamp
        FROM message_add_on_reaction mar
        JOIN message_add_on mao ON mar.message_add_on_row_id = mao._id
        """
        try:
            df = pd.read_sql_query(query, self.conn_msg)
            return df
        except Exception as e:
            print(f"Error parsing reactions: {e}")
            return pd.DataFrame()

    def parse_receipts(self):
        """
        Parses reading receipts.
        Returns DataFrame with: message_row_id, receipt_mode, read_timestamp, played_timestamp
        """
        if not self.conn_msg: return pd.DataFrame()
        
        # Schema for receipts: _id, key_remote_jid, key_id, ...
        # Join with message table to link key_id -> message_row_id
        query = """
        SELECT 
            m._id as message_row_id,
            r.read_device_timestamp as read_timestamp,
            r.played_device_timestamp as played_timestamp
        FROM receipts r
        JOIN message m ON r.key_id = m.key_id
        WHERE r.read_device_timestamp > 0 OR r.played_device_timestamp > 0
        """
        try:
            df = pd.read_sql_query(query, self.conn_msg)
            # Take max timestamp per message (if multiple recipients, this is tricky, 
            # but for 1-1 it's fine. For groups, it implies 'someone' read it).
            # For accurate 1-1 analysis, we assume 1 row per msg usually? 
            # Actually receipts table has one row per recipient per msg.
            # We'll group by message_id and take max read (latest read)
            # or min read (first read). For ghosting, "First Read" is probably better.
            
            # Use min > 0 to find when it was *first* read
            return df
        except Exception as e:
            print(f"Error parsing receipts: {e}")
            return pd.DataFrame()

    def parse_mentions(self):
        if not self.conn_msg: return pd.DataFrame()
        query = """
        SELECT message_row_id, jid_row_id as mentioned_jid_row_id FROM message_mentions
        """
        try:
             return pd.read_sql_query(query, self.conn_msg)
        except: return pd.DataFrame()

    def parse_locations(self):
        if not self.conn_msg: return pd.DataFrame()
        query = """
        SELECT message_row_id, latitude, longitude, place_name, place_address FROM message_location
        """
        try:
             return pd.read_sql_query(query, self.conn_msg)
        except: return pd.DataFrame()

    def get_merged_data(self):
        self.connect()
        
        # 1. Get Messages
        messages_df = self.parse_messages()
        if messages_df.empty:
            return pd.DataFrame()

        # 2. Get JIDs (to link chat_row_id -> raw_string)
        jids_df = self.parse_jids()
        
        # 3. Merge JIDs onto messages
        # Chat JID (Group or 1-on-1 counterpart)
        merged = pd.merge(messages_df, jids_df, on='jid_row_id', how='left')
        
        # 3b. Merge JIDs for SENDER (for Groups)
        # Rename columns to avoid collision
        senders = jids_df.rename(columns={'jid_row_id': 'sender_jid_row_id', 'raw_string': 'sender_string', 'user': 'sender_user'})
        merged = pd.merge(merged, senders, on='sender_jid_row_id', how='left')
        
        # 4. Get Names from WA DB
        wa_contacts_df = self.parse_wa_contacts()
        
        # 5. Get Names from VCF (Dictionary)
        vcf_contacts = self.parse_vcf()
        
        # 6. Resolve Names
        def resolve_name(row):
            # 1. From Me Check
            if row.get('from_me') == 1:
                return "You"
                
            # 2. Determine Target JID (Sender > Chat)
            # In groups, sender_string is the participant. In 1-on-1, it might be null or same.
            target_jid = row.get('sender_string')
            target_user = row.get('sender_user')
            
            if not isinstance(target_jid, str):
                # Fallback to Chat JID (1-on-1)
                target_jid = row.get('raw_string')
                target_user = row.get('user')
                
            if not isinstance(target_jid, str):
                return "Unknown"
            
            # 3. Cleanup JID (remove server suffix for matching)
            # raw_jid looks like 34666123456@s.whatsapp.net
            phone_part = target_jid.split('@')[0]
            
            # 4. Try VCF (Fastest & User Preferred Name)
            if phone_part in vcf_contacts:
                return vcf_contacts[phone_part]
            
            # Fuzzy VCF
            if len(phone_part) > 9:
                suffix = phone_part[-9:]
                if suffix in vcf_contacts:
                    return vcf_contacts[suffix]
                    
            # 5. Try WA DB
            if not wa_contacts_df.empty:
                match = wa_contacts_df[wa_contacts_df['jid'] == target_jid]
                if not match.empty:
                    disp = match.iloc[0]['display_name']
                    if disp: return disp

            # 6. Fallback
            # If it's a group JID (no specific sender identified?), return Subject or "Group"
            if target_jid.endswith("@g.us"):
                return row.get('subject') or "Unknown Group"
                
            return target_user or phone_part

        merged['contact_name'] = merged.apply(resolve_name, axis=1)
        
        # --- Advanced Data Merging ---
        
        # 7. Merge Receipts (for Ghosting)
        receipts = self.parse_receipts()
        if not receipts.empty:
            # Group by message_id, take min read_timestamp > 0
            receipts_clean = receipts[receipts['read_timestamp'] > 0].groupby('message_row_id')['read_timestamp'].min().reset_index()
            # Convert to datetime
            receipts_clean['read_at'] = pd.to_datetime(receipts_clean['read_timestamp'], unit='ms')
            merged = pd.merge(merged, receipts_clean[['message_row_id', 'read_at']], on='message_row_id', how='left')
        else:
            merged['read_at'] = pd.NaT
            
        # 8. Merge Locations
        locs = self.parse_locations()
        if not locs.empty:
            merged = pd.merge(merged, locs, on='message_row_id', how='left')
            
        # 9. Merge Reactions (Aggregation: List of reactions string?)
        # A message can have multiple reactions. We'll store them as a list or string.
        reactions = self.parse_reactions()
        if not reactions.empty:
            # Group to list of reactions per message
            # Also want who reacted? For now, just the reactions themselves for "Most Reacted Msg"
            # Or if we want "Top Reactor", we need the sender logic. 
            # Since 'merged' is per-message, merging 'list of reactions' is best.
            # But calculating "Top Reactor" requires the granular dataframe.
            # We can return the raw reactions DF separately? 
            # OR we can attach a summary to the main DF and logic inside Analyzer can query raw if needed?
            # Current architecture: Parser returns ONE dataframe.
            # Let's attach 'reactions_data' as a JSON-like object or list of tuples? (sender, reaction)
            
            # Let's Resolve Reaction Sender Name
            # We need to map reaction_sender_jid -> Name
            # We can reuse the JID/Contact logic but that's expensive. 
            # Simplified: Map sender_jid_row_id to raw_string using jids_df
            reactions = pd.merge(reactions, jids_df, left_on='sender_jid_row_id', right_on='jid_row_id', how='left')
            # Assuming 'resolve_name' logic can be applied or approximated.
            # For speed, let's just use raw_string or user.
            
            reactions_agg = reactions.groupby('message_row_id').apply(
                lambda x: list(zip(x['reaction'], x['raw_string'] if 'raw_string' in x else x['sender_jid_row_id']))
            ).reset_index(name='reactions_list')
            
            merged = pd.merge(merged, reactions_agg, on='message_row_id', how='left')
        
        # 10. Merge Mentions
        mentions = self.parse_mentions()
        if not mentions.empty:
             # Count mentions per message? Or list of mentioned JIDs?
             mentions_agg = mentions.groupby('message_row_id')['mentioned_jid_row_id'].apply(list).reset_index(name='mentions_list')
             merged = pd.merge(merged, mentions_agg, on='message_row_id', how='left')
             
        return merged

