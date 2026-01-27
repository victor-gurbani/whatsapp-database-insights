import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import os
from wa_analyzer.src.parser import WhatsappParser
from wa_analyzer.src.analyzer import WhatsappAnalyzer
from wordcloud import WordCloud

# Page Config
st.set_page_config(page_title="WhatsApp Analytics", layout="wide", page_icon="💬")

# Title
st.title("💬 WhatsApp Interactive Analyzer")

import json
import datetime

# ... (imports)

# --- Config Management Functions ---
def load_config(uploaded_file):
    if uploaded_file is not None:
        try:
            config = json.load(uploaded_file)
            # Update session state
            for key, value in config.items():
                if key == 'cfg_date_range':
                    # Convert iso strings back to date objects
                    st.session_state[key] = [datetime.date.fromisoformat(d) for d in value]
                else:
                    st.session_state[key] = value
            st.success("Configuration loaded! You may need to wait for reload.")
        except Exception as e:
            st.error(f"Error loading config: {e}")

def get_config_json():
    # Gather keys
    config = {}
    keys_to_save = [
        'cfg_msgstore', 'cfg_wa', 'cfg_vcf', 'cfg_date_range',
        'cfg_ex_groups', 'cfg_ex_chan', 'cfg_ex_system', 'cfg_fam_list', 
        'cfg_ex_fam_glob', 'cfg_ex_non_con', 'cfg_ex_fam_gend', 
        'cfg_long_stats', 'cfg_reply_thresh', 'cfg_min_word_len', 'cfg_ex_emails'
    ]
    for k in keys_to_save:
        if k in st.session_state:
            val = st.session_state[k]
            if isinstance(val, (list, tuple)) and k == 'cfg_date_range':
                 # Serialize dates
                 config[k] = [d.isoformat() for d in val]
            else:
                config[k] = val
    return json.dumps(config, indent=2)

# Sidebar
st.sidebar.header("Data Sources")

# --- Config Import/Export UI (Top of Sidebar for Visibility) ---
with st.sidebar.expander("💾 Configuration Manager", expanded=False):
    uploaded_config = st.file_uploader("Import Config", type=['json'], key="config_uploader")
    if uploaded_config:
        load_config(uploaded_config)
    
    st.divider()
    
    # Export
    # We rely on current session state. Note: Widgets must have keys assigned below.
    json_str = get_config_json()
    st.download_button(
        label="Download Configuration",
        data=json_str,
        file_name="wa_analyzer_config.json",
        mime="application/json"
    )

base_dir = os.getcwd()
# Defaults
default_msgstore = os.path.join(base_dir, "msgstore.db")
default_wa = os.path.join(base_dir, "wa.db")
default_vcf = os.path.join(base_dir, "contacts.vcf")

# Widgets with Keys
msgstore_path = st.sidebar.text_input("Msgstore Path", value=default_msgstore if os.path.exists(default_msgstore) else "", key="cfg_msgstore")
wa_path = st.sidebar.text_input("WA DB Path", value=default_wa if os.path.exists(default_wa) else "", key="cfg_wa")
vcf_path = st.sidebar.text_input("VCF Path", value=default_vcf if os.path.exists(default_vcf) else "", key="cfg_vcf")

if st.sidebar.button("Load Data"):
    with st.spinner("Parsing databases..."):
        parser = WhatsappParser(msgstore_path, wa_path, vcf_path)
        df = parser.get_merged_data()
        
        if df.empty:
            st.error("Failed to parse data or no messages found.")
        else:
            st.session_state['data'] = df
            st.success(f"Loaded {len(df)} messages!")

if 'data' in st.session_state:
    df_raw = st.session_state['data']
    
    # --- Sidebar Filtering ---
    st.sidebar.subheader("Filters")
    
    # Date Filter
    # Ensure default is valid
    if 'cfg_date_range' not in st.session_state:
        min_d = df_raw['timestamp'].min().date()
        max_d = df_raw['timestamp'].max().date()
        st.session_state['cfg_date_range'] = [min_d, max_d]
        
    min_date = df_raw['timestamp'].min().date()
    max_date = df_raw['timestamp'].max().date()
    
    date_range = st.sidebar.date_input(
        "Date Range", 
        value=st.session_state.get('cfg_date_range', [min_date, max_date]), 
        min_value=min_date, 
        max_value=max_date,
        key="cfg_date_range"
    )
    
    # Exclude Groups Filter
    exclude_groups = st.sidebar.checkbox("Exclude Groups", value=False, key="cfg_ex_groups")
    
    # Family Filter
    st.sidebar.subheader("Contact Management")
    all_contacts = sorted(df_raw['contact_name'].unique().astype(str))
    
    # Try to find "Me" or "You" for default selection
    default_fam = []
    if 'cfg_fam_list' not in st.session_state:
        for candidate in ["You", "Me", "Myself", "Tú", "Yo"]:
            match = next((c for c in all_contacts if candidate.lower() == c.lower()), None)
            if match:
                default_fam.append(match)
    else:
        # Pre-fill with what's in session state if available, but ensure it exists in all_contacts
        # Actually standard behavior of multiselect with 'default' is tricky if key exists.
        # If key exists, it overrides 'default'. So we just depend on key.
        pass
            
    family_list = st.sidebar.multiselect(
        "Select Family / Close Contacts", 
        all_contacts, 
        default=default_fam if 'cfg_fam_list' not in st.session_state else None,
        key="cfg_fam_list"
    )
    
    exclude_family_global = st.sidebar.checkbox("Exclude Family from ALL Stats", value=False, key="cfg_ex_fam_glob")
    exclude_non_contacts = st.sidebar.checkbox("Exclude Non-Contacts from ALL Stats", value=False, key="cfg_ex_non_con")
    exclude_channels = st.sidebar.checkbox("Exclude Channels / Announcements", value=True, help="Removes WhatsApp Channels (@newsletter) and Status Broadcasts", key="cfg_ex_chan")
    exclude_system = st.sidebar.checkbox("Exclude System / Security Messages", value=True, help="Removes encryption notices and system markers (Type 7)", key="cfg_ex_system")
    exclude_family_gender = st.sidebar.checkbox("Exclude Family from GENDER Stats Only", value=False, key="cfg_ex_fam_gend")
    
    # Behavioral Config
    st.sidebar.subheader("Behavioral Config")
    use_longer_stats = st.sidebar.checkbox("Use Longer Time Stats", value=False, help="Ghosting: 5 days (vs 24h), Initiation: 2 days (vs 6h)", key="cfg_long_stats")
    
    reply_threshold_hours = st.sidebar.slider(
        "Max Reply Delay (Hours)", 
        min_value=1, 
        max_value=120, 
        value=12,
        help="Messages after this delay are considered new conversations, not replies.",
        key="cfg_reply_thresh"
    )
    
    min_word_len = st.sidebar.number_input(
        "Min Word Length (Word Cloud)", 
        min_value=1, 
        max_value=20, 
        value=4, 
        key="cfg_min_word_len"
    )
    
    exclude_emails = st.sidebar.checkbox("Exclude Emails from Word Cloud", value=False, key="cfg_ex_emails")
    
    # Apply Global Filters
    filtered_df = df_raw.copy()
    
    # Helper for identifying non-contacts (no letters in name)
    import re
    def is_number(name):
        return not bool(re.search('[a-zA-Z]', str(name)))

    if len(date_range) == 2:
        mask = (filtered_df['timestamp'].dt.date >= date_range[0]) & (filtered_df['timestamp'].dt.date <= date_range[1])
        filtered_df = filtered_df.loc[mask]
    if exclude_groups and 'raw_string' in filtered_df.columns:
        is_group = filtered_df['raw_string'].astype(str).str.endswith('@g.us')
        filtered_df = filtered_df[~is_group]
    if exclude_channels and 'raw_string' in filtered_df.columns:
        # Channels usually end in @newsletter. Status is status@broadcast. Official WA is 0@s.whatsapp.net
        is_channel = (
            filtered_df['raw_string'].astype(str).str.endswith('@newsletter') | 
            (filtered_df['raw_string'] == 'status@broadcast') |
            (filtered_df['raw_string'] == '0@s.whatsapp.net')
        )
        filtered_df = filtered_df[~is_channel]
    
    # System Messages (Type 7)
    if exclude_system and 'message_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['message_type'] != 7]
        
    if exclude_family_global and family_list:
        filtered_df = filtered_df[~filtered_df['contact_name'].isin(family_list)]
    if exclude_non_contacts:
        # Remove those that appear to be just numbers
        mask_nums = filtered_df['contact_name'].apply(is_number)
        filtered_df = filtered_df[~mask_nums]
    
    analyzer = WhatsappAnalyzer(filtered_df)

    # --- KPI Row ---
    stats = analyzer.get_basic_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages", f"{stats['total_messages']:,}")
    col2.metric("Sent", f"{stats['sent_count']:,}")
    col3.metric("Received", f"{stats['received_count']:,}")
    col4.metric("Unique Contacts", stats['unique_contacts'])

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Activity & Top Users", "🔥 Behavioral Patterns", "👫 Gender Insights", "📝 Word Cloud", "🔍 Chat Explorer"])

    with tab1:
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.subheader("Top Talkers")
            
            # Filter by Category
            cat_filter = st.selectbox("Filter by Category", ["All", "Only Female", "Only Male", "Only Unknown", "Only Groups", "Only Non-Contacts"])
            
            rank_by_words = st.checkbox("Rank by Total Words", value=False)
            metric_arg = 'words' if rank_by_words else 'messages'
            
            # Fetch a large number first to ensure we fill the top 20 after filtering
            top_talkers_df = analyzer.get_top_talkers(1000, metric=metric_arg) 
            
            if cat_filter == "Only Female":
                top_talkers_df = top_talkers_df[top_talkers_df['gender'] == 'female']
            elif cat_filter == "Only Male":
                top_talkers_df = top_talkers_df[top_talkers_df['gender'] == 'male']
            elif cat_filter == "Only Unknown":
                top_talkers_df = top_talkers_df[top_talkers_df['gender'] == 'unknown']
            elif cat_filter == "Only Groups":
                top_talkers_df = top_talkers_df[top_talkers_df['is_group'] == True]
            elif cat_filter == "Only Non-Contacts":
                mask = top_talkers_df['contact_name'].apply(is_number)
                top_talkers_df = top_talkers_df[mask]
            
            # Slice top 20 after filtering
            top_talkers_final = top_talkers_df.head(20)
            
            fig_bar = px.bar(top_talkers_final, x='count', y='contact_name', orientation='h', 
                             color='gender', title=f"Most Active Contacts ({cat_filter})",
                             color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=600)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_r:
            st.subheader("Hourly Activity")
            split_opt = st.selectbox("Split by:", ["None", "Gender", "Type (Group/Indiv)"], key="hourly_split")
            
            split_arg = None
            if split_opt == "Gender": split_arg = 'gender'
            elif split_opt.startswith("Type"): split_arg = 'group'
            
            hourly = analyzer.get_hourly_activity(split_by=split_arg)
            
            if split_arg:
                fig_line = px.line(hourly, x=hourly.index, y=hourly.columns, markers=True, 
                                   labels={'value': 'Count', 'timestamp': 'Hour'},
                                   title=f"Activity by Hour (Split by {split_opt})")
            else:
                fig_line = px.line(x=hourly.index, y=hourly.values, markers=True, 
                                   labels={'x': 'Hour of Day', 'y': 'Message Count'},
                                   title="Activity by Hour")
            st.plotly_chart(fig_line, use_container_width=True)
            
        st.subheader("Message Volume Over Time")
        show_as_lines = st.checkbox("Show as Lines (Easier Comparison)", value=False)
        plot_func = px.line if show_as_lines else px.area

        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            split_opt_m = st.selectbox("Split Total by:", ["None", "Gender", "Type (Group/Indiv)"], key="monthly_split")
            
            split_arg_m = None
            if split_opt_m == "Gender": split_arg_m = 'gender'
            elif split_opt_m.startswith("Type"): split_arg_m = 'group'
            
            monthly = analyzer.get_monthly_activity(split_by=split_arg_m)
            
            if split_arg_m:
                color_map = {'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'} if split_arg_m == 'gender' else None
                fig_time = plot_func(monthly, x=monthly.index, y=monthly.columns, 
                                     title=f"Total Volume (Split by {split_opt_m})",
                                     color_discrete_map=color_map)
            else:
                fig_time = plot_func(x=monthly.index, y=monthly.values, title="Total Volume")
            st.plotly_chart(fig_time, use_container_width=True)
            
        with col_t2:
            st.write(f"**Top Contacts Volume ({cat_filter})**")
            # Use the already filtered top_talkers_df
            top_contacts_list = top_talkers_final['contact_name'].head(10).tolist()
            if top_contacts_list:
                monthly_contacts = analyzer.get_activity_over_time_by_contact(top_contacts_list)
                fig_contacts = plot_func(monthly_contacts, x=monthly_contacts.index, y=monthly_contacts.columns,
                                       title=f"Top 10 Contacts Activity ({cat_filter})")
                st.plotly_chart(fig_contacts, use_container_width=True)
            else:
                st.info("No contacts match filter.")
        
    with tab2:
        st.header("Behavioral Analysis")
        
        ghost_thresh = 432000 if use_longer_stats else 86400
        init_thresh = 172800 if use_longer_stats else 21600
        
        col_g, col_i = st.columns(2)
        
        with col_g:
            st.subheader("👻 Top Ghosters (Left you on read)")
            st.caption(f"Threshold: {ghost_thresh/3600:.1f} hours silence after your last msg.")
            ghosts = analyzer.get_ghosting_stats(ghost_thresh)
            if not ghosts.empty:
                fig_ghost = px.bar(ghosts, x='count', y='contact_name', orientation='h', 
                                   color='gender', title="Unanswered Threads Count",
                                   color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
                fig_ghost.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_ghost, use_container_width=True)
            else:
                st.write("No ghosting detected!")

            st.divider()
            st.subheader("😶 People I Left on Read")
            ignored = analyzer.get_left_on_read_stats(ghost_thresh)
            if not ignored.empty:
                fig_ignore = px.bar(ignored, x='count', y='contact_name', orientation='h',
                                    color='gender', title="Ignored Threads Count",
                                    color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
                fig_ignore.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_ignore, use_container_width=True)
            else:
                st.write("You reply to everyone 😇")

        with col_i:
            st.subheader("👋 Conversation Initiators")
            st.caption(f"Threshold: {init_thresh/3600:.1f} hours silence.")
            initiations = analyzer.get_initiation_stats(init_thresh)
            if not initiations.empty:
                fig_init = px.bar(initiations[['Me', 'Them']], barmode='group', title="Initiations: Me vs Them")
                st.plotly_chart(fig_init, use_container_width=True)
            else:
                st.write("Not enough data.")
        
        st.divider()
        st.subheader("⏱️ Reply Time Rankings (Avg Minutes)")
        st.caption(f"Minimum 25 messages. Delays > {reply_threshold_hours}h ignored.")
        
        reply_stats = analyzer.get_reply_time_ranking(min_messages=25, max_delay_seconds=reply_threshold_hours*3600)
        
        if not reply_stats.empty:
            rt_col1, rt_col2 = st.columns(2)
            
            # Colors
            color_map = {'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'}
            
            with rt_col1:
                st.write("**Who replies to me the FASTEST?**")
                fastest_them = reply_stats.nsmallest(8, 'their_avg')
                fig_ft = px.bar(fastest_them, x='their_avg', y='contact_name', orientation='h', 
                                color='gender', color_discrete_map=color_map, title="Lowest Avg Reply Time (Them)")
                fig_ft.update_layout(yaxis={'categoryorder':'total descending'}, xaxis_title="Minutes")
                st.plotly_chart(fig_ft, use_container_width=True)
                
                st.write("**Who replies to me the SLOWEST?**")
                slowest_them = reply_stats.nlargest(8, 'their_avg')
                fig_st = px.bar(slowest_them, x='their_avg', y='contact_name', orientation='h', 
                                color='gender', color_discrete_map=color_map, title="Highest Avg Reply Time (Them)")
                fig_st.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Minutes")
                st.plotly_chart(fig_st, use_container_width=True)
                
            with rt_col2:
                st.write("**Who do I reply to the FASTEST?**")
                fastest_me = reply_stats.nsmallest(8, 'my_avg')
                fig_fm = px.bar(fastest_me, x='my_avg', y='contact_name', orientation='h',
                                color='gender', color_discrete_map=color_map, title="My Lowest Avg Reply Time")
                fig_fm.update_layout(yaxis={'categoryorder':'total descending'}, xaxis_title="Minutes")
                st.plotly_chart(fig_fm, use_container_width=True)
                
                st.write("**Who do I reply to the SLOWEST?**")
                slowest_me = reply_stats.nlargest(8, 'my_avg')
                fig_sm = px.bar(slowest_me, x='my_avg', y='contact_name', orientation='h',
                                color='gender', color_discrete_map=color_map, title="My Highest Avg Reply Time")
                fig_sm.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Minutes")
                st.plotly_chart(fig_sm, use_container_width=True)
        else:
            st.info("Not enough conversation data to calculate reply times (need >25 messages).")

    with tab3:
        st.header("Demographics")
        gender_analyzer = analyzer
        if exclude_family_gender and not exclude_family_global and family_list:
             gender_df_source = filtered_df[~filtered_df['contact_name'].isin(family_list)]
             gender_analyzer = WhatsappAnalyzer(gender_df_source)
        
        gender_counts = gender_analyzer.analyze_by_gender()
        gender_stats = gender_analyzer.calculate_gender_stats()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            fig_pie = px.pie(values=gender_counts.values, names=gender_counts.index, 
                             title="Messages by Gender",
                             color=gender_counts.index,
                             color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("Deep Dive Metrics")
            if not gender_stats.empty:
                st.dataframe(gender_stats, use_container_width=True)
                metrics = ['count', 'avg_wpm', 'media_pct', 'avg_reply_time']
                metric_choice = st.selectbox("Select Metric", metrics)
                fig_comp = px.bar(gender_stats, x='gender', y=metric_choice, 
                                  color='gender', title=f"Comparison: {metric_choice}",
                                  color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
                st.plotly_chart(fig_comp, use_container_width=True)

    with tab4:
        st.header("Word Cloud")
        
        wc_source = st.radio("Source", ["All Messages", "Sent by Me", "Received by Me"], horizontal=True)
        
        if st.button("Generate Word Cloud"):
            with st.spinner("Generating..."):
                filter_me = None
                if wc_source == "Sent by Me": filter_me = True
                elif wc_source == "Received by Me": filter_me = False
                
                text = analyzer.get_wordcloud_text(filter_from_me=filter_me, min_word_length=min_word_len, exclude_emails=exclude_emails)
                if text:
                    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wc, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                else:
                    st.warning("Not enough text data.")

    with tab5:
        st.header("Chat Explorer & Deep Dive")
        contacts = sorted(filtered_df['contact_name'].unique().astype(str))
        selected_contact = st.selectbox("Select Contact", contacts)
        
        if selected_contact:
            sub_df = filtered_df[filtered_df['contact_name'] == selected_contact]
            st.write(f"### Analysis: **{selected_contact}**")
            st.write(f"Total Messages: {len(sub_df)}")
            
            chat_analyzer = WhatsappAnalyzer(sub_df)
            my_reply, their_reply = chat_analyzer.calculate_chat_reply_times()
            
            col_s1, col_s2 = st.columns(2)
            col_s1.metric("My Avg Reply Time", f"{my_reply:.1f} min")
            col_s2.metric("Their Avg Reply Time", f"{their_reply:.1f} min")
            
            st.subheader("Behavioral Timeline")
            # Get behavioral timeline data
            ghost_thresh = 432000 if use_longer_stats else 86400
            init_thresh = 172800 if use_longer_stats else 21600
            
            beh_timeline = chat_analyzer.get_behavioral_timeline(ghost_thresh, init_thresh)
            
            b1, b2 = st.columns(2)
            with b1:
                st.caption("Ghosting Over Time")
                # Ghosted by Them vs Ghosted by Me
                fig_g = px.bar(beh_timeline, x=beh_timeline.index, y=['Ghosted by Them', 'Ghosted by Me'], 
                               title="Ghosting Incidents", barmode='group')
                st.plotly_chart(fig_g, use_container_width=True)
                
            with b2:
                st.caption("Initiations Over Time")
                fig_i = px.bar(beh_timeline, x=beh_timeline.index, y=['Initiated by Me', 'Initiated by Them'], 
                               title="Conversation Initiations", barmode='group')
                st.plotly_chart(fig_i, use_container_width=True)

            st.subheader("Activity Analysis")
            
            # Controls for Activity Logic
            c_act1, c_act2 = st.columns([2, 1])
            with c_act1:
                act_view = st.radio("View Mode", ["Combined", "Split (Me vs Them)", "Only Me", "Only Them"], horizontal=True, key="chat_act_view")
            with c_act2:
                chat_show_lines = st.checkbox("Show as Lines", value=False, key="chat_show_lines")
            
            plot_func_chat = px.line if chat_show_lines else px.area

            # Monthly Activity
            monthly_split_arg = 'sender' if act_view != "Combined" else None
            monthly_chat = chat_analyzer.get_monthly_activity(split_by=monthly_split_arg)
            
            # Filter if needed
            if act_view == "Only Me" and 'Me' in monthly_chat.columns:
                monthly_chat = monthly_chat[['Me']]
            elif act_view == "Only Them" and 'Them' in monthly_chat.columns:
                monthly_chat = monthly_chat[['Them']]
            
            if isinstance(monthly_chat, pd.DataFrame):
                fig_chat_time = plot_func_chat(monthly_chat, x=monthly_chat.index, y=monthly_chat.columns, title=f"Message Volume ({act_view})")
            else:
                fig_chat_time = plot_func_chat(x=monthly_chat.index, y=monthly_chat.values, title=f"Message Volume ({act_view})")
            st.plotly_chart(fig_chat_time, use_container_width=True)
            
            # Hourly Activity
            hourly_split_arg = 'sender' if act_view != "Combined" else None
            hourly_chat = chat_analyzer.get_hourly_activity(split_by=hourly_split_arg)
            
            if isinstance(hourly_chat, pd.DataFrame):
                if act_view == "Only Me" and 'Me' in hourly_chat.columns:
                    hourly_chat = hourly_chat[['Me']]
                elif act_view == "Only Them" and 'Them' in hourly_chat.columns:
                    hourly_chat = hourly_chat[['Them']]
                fig_chat_hour = px.line(hourly_chat, x=hourly_chat.index, y=hourly_chat.columns, markers=True, title=f"Hourly Activity ({act_view})")
            else:
                fig_chat_hour = px.line(x=hourly_chat.index, y=hourly_chat.values, markers=True, title=f"Hourly Activity ({act_view})")
                
            st.plotly_chart(fig_chat_hour, use_container_width=True)

            st.subheader("Word Usage Comparison")
            if st.button("Generate Comparative Word Clouds"):
                col_wc1, col_wc2 = st.columns(2)
                with col_wc1:
                    st.caption("My Words")
                    my_text = chat_analyzer.get_wordcloud_text(filter_from_me=True, min_word_length=min_word_len, exclude_emails=exclude_emails)
                    if my_text:
                        wc1 = WordCloud(width=400, height=300, background_color='white').generate(my_text)
                        plt.figure(figsize=(5,4))
                        plt.imshow(wc1)
                        plt.axis('off')
                        st.pyplot(plt)
                    else: st.write("No data.")
                        
                with col_wc2:
                    st.caption(f"{selected_contact}'s Words")
                    their_text = chat_analyzer.get_wordcloud_text(filter_from_me=False, min_word_length=min_word_len, exclude_emails=exclude_emails)
                    if their_text:
                        wc2 = WordCloud(width=400, height=300, background_color='white').generate(their_text)
                        plt.figure(figsize=(5,4))
                        plt.imshow(wc2)
                        plt.axis('off')
                        st.pyplot(plt)
                    else: st.write("No data.")
 
            st.write("Recent Messages:")
            cols_to_show = ['timestamp', 'from_me', 'text_data']
            if 'mime_type' in sub_df.columns: cols_to_show.append('mime_type')
            st.dataframe(sub_df[cols_to_show].sort_values('timestamp', ascending=False).head(10))

else:
    st.info("👈 Please enter file paths.")
