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
# --- Config Management Functions ---
def load_config():
    uploaded_file = st.session_state.get('config_uploader')
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
            # st.rerun() not needed in callback
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
    return json.dumps(config, indent=2, sort_keys=True)

# Sidebar
st.sidebar.header("Data Sources")

# --- Config Import/Export UI (Top of Sidebar for Visibility) ---
with st.sidebar.expander("💾 Configuration Manager", expanded=False):
    st.file_uploader("Import Config", type=['json'], key="config_uploader", on_change=load_config)
    
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
    min_date = df_raw['timestamp'].min().date()
    max_date = df_raw['timestamp'].max().date()
    
    # We pass the default value here. If 'cfg_date_range' is already in session_state 
    # (e.g. from loaded config), Streamlit will use that value instead.
    # We do NOT manually set session_state['cfg_date_range'] here to avoid conflicts.
    
    # Quick Date Buttons
    st.sidebar.caption("Quick Date Filters")
    cols_q = st.sidebar.columns(5)
    labels = ["3M", "6M", "1Y", "3Y", "10Y"]
    offsets = [3, 6, 12, 36, 120] # Months
    
    for i, label in enumerate(labels):
        if cols_q[i].button(label):
            new_start = max_date - pd.DateOffset(months=offsets[i])
            # Convert to date object because date_input expects dates
            new_start = new_start.date()
            if new_start < min_date: new_start = min_date
            st.session_state['cfg_date_range'] = [new_start, max_date]
            st.rerun()

    if st.sidebar.button("Reset Date"):
         st.session_state['cfg_date_range'] = [min_date, max_date]
         st.rerun()
    
    date_range = st.sidebar.date_input(
        "Date Range", 
        value=[min_date, max_date], 
        min_value=min_date, 
        max_value=max_date,
        key="cfg_date_range"
    )
    
    # --- Identity Status ---
    st.sidebar.markdown("### 👤 Identity Status")
    n_sent = len(df_raw[df_raw['from_me'] == 1])
    n_recv = len(df_raw[df_raw['from_me'] == 0])
    st.sidebar.caption(f"Detected **{n_sent:,}** sent and **{n_recv:,}** received messages.")
    st.sidebar.divider()
    
    # Exclude Groups Filter
    exclude_groups = st.sidebar.checkbox("Exclude Groups", value=False, key="cfg_ex_groups")
    
    # Exclude Me Filter
    exclude_me = st.sidebar.checkbox("Exclude 'Me/You' from Charts", value=True, key="cfg_ex_me", help="Remove your own sent messages from Activity and Top Talkers charts.")
    
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
    exclude_family_behavior = st.sidebar.checkbox("Exclude Family from BEHAVIORAL Stats", value=False, key="cfg_ex_fam_beh")
    
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
    # 1. Base Filters (Used for Chat Explorer & Identity)
    df_base = df_raw.copy()
    
    # Helper for identifying non-contacts (no letters in name)
    import re
    def is_number(name):
        return not bool(re.search('[a-zA-Z]', str(name)))

    # SECURITY MESSAGES (ALWAYS EXCLUDE)
    if 'message_type' in df_base.columns:
        df_base = df_base[df_base['message_type'] != 7]

    if len(date_range) == 2:
        mask = (df_base['timestamp'].dt.date >= date_range[0]) & (df_base['timestamp'].dt.date <= date_range[1])
        df_base = df_base.loc[mask]
    if exclude_groups and 'raw_string' in df_base.columns:
        is_group = df_base['raw_string'].astype(str).str.endswith('@g.us')
        df_base = df_base[~is_group]
    if exclude_channels and 'raw_string' in df_base.columns:
        # Channels usually end in @newsletter. Status is status@broadcast. Official WA is 0@s.whatsapp.net
        is_channel = (
            df_base['raw_string'].astype(str).str.endswith('@newsletter') | 
            (df_base['raw_string'] == 'status@broadcast') |
            (df_base['raw_string'] == '0@s.whatsapp.net')
        )
        df_base = df_base[~is_channel]
    
    # System Messages (Type 7) - Already excluded above unconditionally
    # if exclude_system and 'message_type' in filtered_df.columns:
    #     filtered_df = filtered_df[filtered_df['message_type'] != 7]
        
    if exclude_family_global and family_list:
        df_base = df_base[~df_base['chat_name'].isin(family_list)]
    if exclude_non_contacts:
        # Remove those that appear to be just numbers
        mask_nums = df_base['contact_name'].apply(is_number)
        df_base = df_base[~mask_nums]

    # 2. View Filters (Used for General Stats where Me skews it)
    filtered_df = df_base.copy()
    if exclude_me:
        filtered_df = filtered_df[filtered_df['from_me'] == 0]

    # Update Identity Info with Phone Number attempt
    me_jid = None
    me_rows = df_raw[df_raw['from_me'] == 1]
    if not me_rows.empty:
        possible = me_rows['sender_string'].dropna().unique()
        if len(possible) > 0:
            me_jid = possible[0]
    
    me_display = me_jid.split('@')[0] if me_jid else "Unknown"
        
    st.sidebar.markdown(f"**User**: {me_display}")

    analyzer = WhatsappAnalyzer(filtered_df)
    
    analyzer = WhatsappAnalyzer(filtered_df)

    # --- KPI Row ---
    stats = analyzer.get_basic_stats()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages", f"{stats['total_messages']:,}")
    col2.metric("Sent", f"{stats['sent_count']:,}")
    col3.metric("Received", f"{stats['received_count']:,}")
    col4.metric("Unique Contacts", stats['unique_contacts'])

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Activity & Top Users", "🔥 Behavioral Patterns", "👫 Gender Insights", "📝 Word Cloud", "🔍 Chat Explorer", "🎪 Fun & Insights", "🗺️ Map"])

    with tab1:
        col_l, col_r = st.columns(2)
        
        with col_l:
            st.subheader("Top Talkers")
            
            # Filter by Category
            cat_filter = st.selectbox("Filter by Category", ["All", "Only Female", "Only Male", "Only Unknown", "Only Groups", "Only Non-Contacts"])
            
            rank_by_words = st.checkbox("Rank by Total Words", value=False)
            metric_arg = 'words' if rank_by_words else 'messages'
            
            # Fetch a large number first to ensure we fill the top 20 after filtering
            top_talkers_df = analyzer.get_top_talkers(1000, metric=metric_arg, exclude_me=exclude_me) 
            
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
            st.plotly_chart(fig_bar, width='stretch')
            
        with col_r:
            st.subheader("Hourly Activity")
            split_opt = st.selectbox("Split by:", ["None", "Gender", "Type (Group/Indiv)"], key="hourly_split")
            
            split_arg = None
            if split_opt == "Gender": split_arg = 'gender'
            elif split_opt.startswith("Type"): split_arg = 'group'
            
            hourly = analyzer.get_hourly_activity(split_by=split_arg, exclude_me=exclude_me)
            
            if split_arg:
                fig_line = px.line(hourly, x=hourly.index, y=hourly.columns, markers=True, 
                                   labels={'value': 'Count', 'timestamp': 'Hour'},
                                   title=f"Activity by Hour (Split by {split_opt})")
            else:
                fig_line = px.line(x=hourly.index, y=hourly.values, markers=True, 
                                   labels={'x': 'Hour of Day', 'y': 'Message Count'},
                                   title="Activity by Hour")
            st.plotly_chart(fig_line, width='stretch')
            
        st.subheader("Message Volume Over Time")
        show_as_lines = st.checkbox("Show as Lines (Easier Comparison)", value=False)
        plot_func = px.line if show_as_lines else px.area

        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            split_opt_m = st.selectbox("Split Total by:", ["None", "Gender", "Type (Group/Indiv)"], key="monthly_split")
            
            split_arg_m = None
            if split_opt_m == "Gender": split_arg_m = 'gender'
            elif split_opt_m.startswith("Type"): split_arg_m = 'group'
            
            monthly = analyzer.get_monthly_activity(split_by=split_arg_m, exclude_me=exclude_me)
            
            if split_arg_m:
                color_map = {'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'} if split_arg_m == 'gender' else None
                fig_time = plot_func(monthly, x=monthly.index, y=monthly.columns, 
                                     title=f"Total Volume (Split by {split_opt_m})",
                                     color_discrete_map=color_map)
            else:
                fig_time = plot_func(x=monthly.index, y=monthly.values, title="Total Volume")
            st.plotly_chart(fig_time, width='stretch')
            
        with col_t2:
            st.write(f"**Top Contacts Volume ({cat_filter})**")
            # Use the already filtered top_talkers_df
            top_contacts_list = top_talkers_final['contact_name'].head(10).tolist()
            if top_contacts_list:
                monthly_contacts = analyzer.get_activity_over_time_by_contact(top_contacts_list)
                fig_contacts = plot_func(monthly_contacts, x=monthly_contacts.index, y=monthly_contacts.columns,
                                       title=f"Top 10 Contacts Activity ({cat_filter})")
                st.plotly_chart(fig_contacts, width='stretch')
            else:
                st.info("No contacts match filter.")
        
    with tab2:
        st.header("Behavioral Analysis")
        
        # USE FULL ANALYZER (Includes 'Me') for interactions
        # Because we need to know if 'I' replied or initiated.
        full_analyzer = WhatsappAnalyzer(df_base)
        
        ghost_thresh = 432000 if use_longer_stats else 86400
        init_thresh = 172800 if use_longer_stats else 21600
        
        col_g, col_i = st.columns(2)
        
        with col_g:
            st.subheader("👻 Top Ghosters (Left you on read)")
            st.caption(f"Threshold: {ghost_thresh/3600:.1f} hours silence after your last msg.")
            st.info("ℹ️ 'End of Data' Logic: If a conversation extends to the very end of your message history (e.g. yesterday), it is NOT counted as ghosting.")
            
            bhv_exclude = family_list if exclude_family_behavior else None
            
            ghosts = full_analyzer.get_ghosting_stats(ghost_thresh, exclude_list=bhv_exclude)
            if not ghosts.empty:
                fig_ghost = px.bar(ghosts, x='count', y='contact_name', orientation='h', 
                                   color='gender', title="Unanswered Threads Count",
                                   color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
                fig_ghost.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_ghost, width='stretch')
            else:
                st.write("No ghosting detected!")

            st.divider()
            st.subheader("😶 People I Ignore (Me → Them)")
            ignored = full_analyzer.get_left_on_read_stats(ghost_thresh, exclude_list=bhv_exclude)
            if not ignored.empty:
                # ignored is a pivot table with columns like 'True Ghost', 'Left on Delivered', 'Total Ignored'
                # We can plot 'Total Ignored' or stack the types. Stacked is better.
                # Reset index to get contact_name as column
                ignored_reset = ignored.reset_index()
                cols_to_plot = [c for c in ignored.columns if c in ['True Ghost 👻', 'Left on Delivered 📨']]
                
                # Check if ignored is empty (it might have gender but no counts if all 0, but filtering logic in analyzer handles non-empty)
                
                # Check for gender column (added in latest update)
                color_arg = 'gender' if 'gender' in ignored_reset.columns else None
                color_map = {'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'} if color_arg else None

                fig_ignore = px.bar(ignored_reset, x=cols_to_plot, y='contact_name', orientation='h',
                                    title="Ignored Threads Count",
                                    color=color_arg, color_discrete_map=color_map)
                fig_ignore.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Count")
                st.plotly_chart(fig_ignore, width='stretch')
            else:
                st.write("You reply to everyone 😇")

        with col_i:
            st.subheader("👋 Conversation Initiators")
            st.caption(f"Threshold: {init_thresh/3600:.1f} hours silence.")
            initiations = full_analyzer.get_initiation_stats(init_thresh, exclude_list=bhv_exclude)
            if not initiations.empty:
                fig_init = px.bar(initiations[['Me', 'Them']], barmode='group', title="Initiations: Me vs Them")
                st.plotly_chart(fig_init, width='stretch')
            else:
                st.write("Not enough data.")
        
        st.divider()
        st.subheader("⏱️ Reply Time Rankings (Avg Minutes)")
        
        rc1, rc2 = st.columns(2)
        min_msgs_input = rc1.number_input("Min Messages", min_value=5, value=25, step=5)
        top_30_only = rc2.checkbox("Rank Only Top 30 Contacts", value=True)
        
        st.caption(f"Delays > {reply_threshold_hours}h ignored.")
        
        # We need the FULL transaction history (including ME) to calculate reply times.
        # 'analyzer' uses filtered_df which might exclude 'Me'.
        full_analyzer = WhatsappAnalyzer(df_base)
        
        reply_stats = full_analyzer.get_reply_time_ranking(min_messages=min_msgs_input, max_delay_seconds=reply_threshold_hours*3600, exclude_list=bhv_exclude)
        
        if top_30_only and not reply_stats.empty:
            # Get Top 30 names
            top_30 = analyzer.get_top_talkers(30)['contact_name'].tolist()
            reply_stats = reply_stats[reply_stats['contact_name'].isin(top_30)]
        
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
                st.plotly_chart(fig_ft, width='stretch')
                
                st.write("**Who replies to me the SLOWEST?**")
                slowest_them = reply_stats.nlargest(8, 'their_avg')
                fig_st = px.bar(slowest_them, x='their_avg', y='contact_name', orientation='h', 
                                color='gender', color_discrete_map=color_map, title="Highest Avg Reply Time (Them)")
                fig_st.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Minutes")
                st.plotly_chart(fig_st, width='stretch')
                
            with rt_col2:
                st.write("**Who do I reply to the FASTEST?**")
                fastest_me = reply_stats.nsmallest(8, 'my_avg')
                fig_fm = px.bar(fastest_me, x='my_avg', y='contact_name', orientation='h',
                                color='gender', color_discrete_map=color_map, title="My Lowest Avg Reply Time")
                fig_fm.update_layout(yaxis={'categoryorder':'total descending'}, xaxis_title="Minutes")
                st.plotly_chart(fig_fm, width='stretch')
                
                st.write("**Who do I reply to the SLOWEST?**")
                slowest_me = reply_stats.nlargest(8, 'my_avg')
                fig_sm = px.bar(slowest_me, x='my_avg', y='contact_name', orientation='h',
                                color='gender', color_discrete_map=color_map, title="My Highest Avg Reply Time")
                fig_sm.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Minutes")
                st.plotly_chart(fig_sm, width='stretch')
        else:
            st.info("Not enough conversation data to calculate reply times (need >25 messages).")

    with tab3:
        st.header("Demographics")
        # Use full_analyzer to ensure Reply Time calc has 'Me' messages
        # analyze_by_gender() handles 'from_me=0' internally for volume.
        gender_analyzer = WhatsappAnalyzer(df_base)
        
        if exclude_family_gender and not exclude_family_global and family_list:
             # Apply family filter to base for gender analysis
             # Note: df_base includes Me. gender_stats needs Me for reply time.
             gender_df_source = df_base[~df_base['chat_name'].isin(family_list)]
             gender_analyzer = WhatsappAnalyzer(gender_df_source)
        
        gender_counts = gender_analyzer.analyze_by_gender()
        gender_stats = gender_analyzer.calculate_gender_stats()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            fig_pie = px.pie(values=gender_counts.values, names=gender_counts.index, 
                             title="Messages by Gender",
                             color=gender_counts.index,
                             color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
            st.plotly_chart(fig_pie, width='stretch')
        with c2:
            st.subheader("Deep Dive Metrics")
            if not gender_stats.empty:
                st.dataframe(gender_stats, width='stretch')
                metrics = ['count', 'avg_wpm', 'media_pct', 'avg_reply_time']
                metric_choice = st.selectbox("Select Metric", metrics)
                fig_comp = px.bar(gender_stats, x='gender', y=metric_choice, 
                                  color='gender', title=f"Comparison: {metric_choice}",
                                  color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
                st.plotly_chart(fig_comp, width='stretch')

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
            # Use df_base (includes 'Me') and filter by chat_name to get the full conversation
            sub_df = df_base[df_base['chat_name'] == selected_contact]
            st.write(f"### Analysis: **{selected_contact}**")
            st.write(f"Total Messages: {len(sub_df)}")
            
            chat_analyzer = WhatsappAnalyzer(sub_df)
            my_reply, their_reply = chat_analyzer.calculate_chat_reply_times()
            
            col_s1, col_s2 = st.columns(2)
            col_s1.metric("My Avg Reply Time", f"{my_reply:.1f} min")
            col_s2.metric("Their Avg Reply Time", f"{their_reply:.1f} min")
            
            # --- Advanced Chat Stats ---
            # Use chat_analyzer to get stats for this specific chat (efficient & correct context)
            dist_them, _ = chat_analyzer.get_advanced_reply_stats(reply_to=0)
            dist_me, _ = chat_analyzer.get_advanced_reply_stats(reply_to=1)
            
            st.write("### Response Time Analysis")
            
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.write("**Their Speed (Them → Me)**")
                if dist_them is not None and selected_contact in dist_them.index:
                    row = dist_them.loc[selected_contact]
                    fig_dist = px.bar(x=row.index, y=row.values, labels={'x': 'Time', 'y': 'Count'}, title=f"{selected_contact}'s Speed")
                    st.plotly_chart(fig_dist, use_container_width=True)
                else: st.caption("No data")
                    
            with col_d2:
                st.write("**My Speed (Me → Them)**")
                if dist_me is not None and selected_contact in dist_me.index:
                    row_me = dist_me.loc[selected_contact]
                    fig_dist_me = px.bar(x=row_me.index, y=row_me.values, labels={'x': 'Time', 'y': 'Count'}, title="My Speed")
                    fig_dist_me.update_traces(marker_color='#EF553B')
                    st.plotly_chart(fig_dist_me, use_container_width=True)
                
            # Ghosting Control
            st.divider()
            gh_hours = st.slider("Ghosting Threshold (Hours)", 1, 72, 24, key=f"gh_{selected_contact}")
            true_ghosts = analyzer.get_true_ghosting_stats(threshold_hours=gh_hours)
            if not true_ghosts.empty and selected_contact in true_ghosts.index:
                st.write(f"**Ghosting Stats (> {gh_hours}h)**")
                g_row = true_ghosts.loc[selected_contact]
                cols_g = st.columns(3)
                cols_g[0].metric("True Ghosts 👻", int(g_row.get('True Ghost 👻', 0)), help="Read but ignored")
                cols_g[1].metric("Left on Delivered 📨", int(g_row.get('Left on Delivered 📨', 0)), help="Never read")
            else:
                st.info("No ghosting detected with current threshold.")
            
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
                st.plotly_chart(fig_g, width='stretch')
                
            with b2:
                st.caption("Initiations Over Time")
                fig_i = px.bar(beh_timeline, x=beh_timeline.index, y=['Initiated by Me', 'Initiated by Them'], 
                               title="Conversation Initiations", barmode='group')
                st.plotly_chart(fig_i, width='stretch')

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
            st.plotly_chart(fig_chat_time, width='stretch')
            
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
                
            st.plotly_chart(fig_chat_hour, width='stretch')

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

    with tab6:
        st.header("🎪 Fun & Insights")
        
        col_fun_ctrl1, col_fun_ctrl2 = st.columns(2)
        top_n_filter = col_fun_ctrl1.selectbox("Filter Rank to Top Contacts:", [50, 100, 200, "All"], index=1, help="Analyze only the most active contacts to reduce noise.")
        top_n_val = None if top_n_filter == "All" else int(top_n_filter)
        
        
        # Calculate Stats
        # Pass exclude_groups from sidebar
        # Calculate Stats using FULL Data (Context needed for Double Text, Streaks, Killers)
        # But we must respect exclude_me for the DISPLAY.
        full_analyzer_tab6 = WhatsappAnalyzer(df_base)
        
        # Pass exclude_groups from sidebar
        ex_groups = exclude_groups if 'exclude_groups' in locals() else False
        
        beh_scorecard = full_analyzer_tab6.get_behavioral_scorecard(exclude_groups=ex_groups) 
        fun_stats = full_analyzer_tab6.get_fun_stats(top_n=top_n_val, exclude_groups=ex_groups)
        streaks = full_analyzer_tab6.get_streak_stats(exclude_groups=ex_groups)
        killers = full_analyzer_tab6.get_conversation_killers(exclude_groups=ex_groups)
        
        # New Stats
        reaction_stats = full_analyzer_tab6.get_reaction_stats(exclude_groups=ex_groups)
        emoji_stats = full_analyzer_tab6.get_emoji_stats(top_n=top_n_val, exclude_groups=ex_groups)
        mention_stats = full_analyzer_tab6.get_mention_stats(top_n=top_n_val, exclude_groups=ex_groups)
        history_stats = full_analyzer_tab6.get_historical_stats(exclude_groups=ex_groups)
        
        # Post-Calculation Filter: Remove "You" / "Me" if exclude_me is True
        if exclude_me:
             # Identify 'Me' name (usually 'You', 'Me', 'Myself' or me_display)
             # Parser ensures outgoing is mapped to "You". 
             me_names = ['You', 'Me', 'Myself']
             if 'me_display' in locals(): me_names.append(str(me_display)) # if resolved
             
             # Filter Index (usually contact_name)
             if not beh_scorecard.empty: beh_scorecard = beh_scorecard[~beh_scorecard.index.isin(me_names)]
             if not fun_stats.empty: fun_stats = fun_stats[~fun_stats.index.isin(me_names)]
             if not streaks.empty: streaks = streaks[~streaks.index.isin(me_names)]
             if not killers.empty: killers = killers[~killers.index.isin(me_names)]
             
             # Reaction stats is a dict
             if reaction_stats and 'top_reactors' in reaction_stats:
                  reaction_stats['top_reactors'] = reaction_stats['top_reactors'][~reaction_stats['top_reactors'].index.isin(me_names)]
             
             # Emoji stats is dict
             # per_contact...
             if emoji_stats and 'per_contact' in emoji_stats:
                  # per_contact has 'contact_name' column
                  emoji_stats['per_contact'] = emoji_stats['per_contact'][~emoji_stats['per_contact']['contact_name'].isin(me_names)]
             
             # Mention stats is dict
             # who_mentions_me (index is sender)
             if mention_stats and 'who_mentions_me' in mention_stats:
                  mention_stats['who_mentions_me'] = mention_stats['who_mentions_me'][~mention_stats['who_mentions_me'].index.isin(me_names)]
             # i_mention (index is target) - keep? Yes, it's who I mention.

        
        # Pre-calc combined media for Gallery Curator
        if 'image_media' in fun_stats.columns and 'video_media' in fun_stats.columns:
            fun_stats['gallery_count'] = fun_stats['image_media'] + fun_stats['video_media']
        else:
            fun_stats['gallery_count'] = fun_stats['media'] # Fallback
            
        # 1. Hall of Fame
        st.subheader("🏆 Hall of Fame")
        hof_1, hof_2, hof_3 = st.columns(3)
        
        # Helper to get top user
        def get_top(df, col, exclude_list=[]):
            if df.empty or col not in df.columns: return "N/A", 0
            # Filter
            idx = df.index
            # Exclude numbers if possible
            # We already have filters applied at 'analyzer' level but let's double check
            sorted_df = df.sort_values(col, ascending=False)
            if sorted_df.empty: return "N/A", 0
            top_name = sorted_df.index[0]
            top_val = sorted_df[col].iloc[0]
            return top_name, top_val

        with hof_1:
            name, val = get_top(beh_scorecard, 'night_owl_pct')
            st.metric("🦉 The Night Owl", name, f"{val:.1f}% Night Msgs")
            
        with hof_2:
            name, val = get_top(beh_scorecard, 'early_bird_pct')
            st.metric("☀️ The Early Bird", name, f"{val:.1f}% Morning Msgs")
            
        with hof_3:
            name, val = get_top(fun_stats, 'laughs')
            st.metric("😂 The Comedian", name, f"{int(val)} Laughs")
            
        st.divider()
        
        hof_4, hof_5, hof_6 = st.columns(3)
        with hof_4:
            name, val = get_top(fun_stats, 'deleted')
            st.metric("🗑️ The Deleter", name, f"{int(val)} Retracted")
            
        with hof_5:
            if not streaks.empty:
                name = streaks.idxmax()
                val = streaks.max()
            else: name, val = "N/A", 0
            st.metric("🔥 Streak Master", name, f"{val} Days")
            
            if not killers.empty:
                name = killers.index[0]
                val = killers.iloc[0]
            else: name, val = "N/A", 0
            st.metric("🤐 Conversation Killer", name, f"{val} Silences (>24h)")
            
        st.divider()
        
        hof_7, hof_8, hof_9 = st.columns(3)
        with hof_7:
            name, val = get_top(fun_stats, 'audio_media')
            st.metric("🎙️ The Podcaster", name, f"{int(val)} Voice Notes")
            
        with hof_8:
            name, val = get_top(fun_stats, 'gallery_count')
            st.metric("🖼️ Gallery Curator", name, f"{int(val)} Pics/Vids")
            
        with hof_9:
             if reaction_stats and not reaction_stats['top_reactors'].empty:
                 name = reaction_stats['top_reactors'].index[0]
                 val = reaction_stats['top_reactors'].iloc[0]
             else: name, val = "N/A", 0
             st.metric("😍 Reaction addict", name, f"{val} Reactions")
             
        st.divider()
        
        # --- NEW SECTIONS ---
        
        # 1. Emoji Analysis
        if emoji_stats and not emoji_stats['per_contact'].empty:
             st.subheader("❤️ The Emoji Fanatic")
             # Show Top 5 Emojis for Top 5 Users
             # emoji_stats['per_contact'] is a DF with contact, emoji, count
             # Pivot to allow nice display? Or just list?
             # Let's show a dataframe of "User | Top 5 Emojis"
             
             # Group by contact, join emojis
             # --- UI IMPROVEMENT: Default Top 10 + Search ---
             all_contacts_emoji = emoji_stats['per_contact']['contact_name'].unique().tolist()
             default_emoji_sel = all_contacts_emoji[:10]
             
             sel_emoji_contacts = st.multiselect("Select Contacts", all_contacts_emoji, default=default_emoji_sel)
             
             if sel_emoji_contacts:
                 filtered_emoji = emoji_stats['per_contact'][emoji_stats['per_contact']['contact_name'].isin(sel_emoji_contacts)]
                 top_emo_disp = filtered_emoji.groupby('contact_name')['emoji'].apply(lambda x: " ".join(x)).reset_index(name='Top Emojis')
                 st.dataframe(top_emo_disp.set_index('contact_name'), use_container_width=True)
             else:
                 st.write("No contacts selected.")
             
        # 2. Reaction Deep Dive
        if reaction_stats:
            st.subheader("😍 Reaction Insights")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.write("**Most Reacted Messages**")
                st.dataframe(reaction_stats['most_reacted'][['chat_contact', 'preview', 'count']].head(5))
            with col_r2:
                st.write("**Top Global Emojis**")
                # Top emojis as bar chart
                top_em = reaction_stats['top_emojis']
                st.bar_chart(top_em)
                
        # 3. Mentions
        if mention_stats:
            st.subheader("📢 Mentions (@Tags)")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.write("**Who Mentions Me Most?**")
                st.bar_chart(mention_stats['who_mentions_me'])
            with col_m2:
                st.write("**Who Do I Mention Most?**")
                st.bar_chart(mention_stats['i_mention'])
                
        # 4. Historical Deep Dive
        if history_stats:
            st.subheader("📜 Historical Deep Dive")
            
            # Velocity
            st.write("**⚡ Message Velocity (Max Words Per Minute)**")
            st.caption("Highest WPM achieved in a single minute of conversation.")
            
            # --- UI IMPROVEMENT: Default Top 10 + Search ---
            velocity_df = history_stats['velocity_wpm'] # Series
            all_vel_contacts = velocity_df.index.tolist()
            default_vel = all_vel_contacts[:10]
            
            sel_vel_contacts = st.multiselect("Select Contacts for Velocity", all_vel_contacts, default=default_vel)
            
            if sel_vel_contacts:
                filtered_vel = velocity_df[velocity_df.index.isin(sel_vel_contacts)]
                st.bar_chart(filtered_vel, horizontal=True)
            else:
                st.write("No contacts selected.")
            
            # First Message
            # First Message
            st.write("**🕰️ How it Started (First Messages)**")
            first_msgs = history_stats['first_msgs'].reset_index()
            # Select contact
            sel_contact_hist = st.selectbox("Select Contact to see First Message:", first_msgs['chat_name'].unique())
            if sel_contact_hist:
                row = first_msgs[first_msgs['chat_name'] == sel_contact_hist].iloc[0]
                st.markdown(f"**Date:** {row['timestamp']}")
                st.markdown(f"**Message:** *\"{row['text_data']}\"*")

        st.divider()
        col_lod1, col_lod2 = st.columns(2)
        
        with col_lod1:
             st.write("**People I Ignore** (Me → Them)")
             # Use full_analyzer to capture 'Me' messages even if view is excluded
             my_ignore_stats = full_analyzer_tab6.get_left_on_read_stats()
             
             # Post-calc filter: Remove 'Me' from the list of people I ignore?
             # No, the list contains contacts I ignore. 'Me' shouldn't be there.
             # But if exclude_me is True, we should ensure 'Me' isn't in index if it somehow got there.
             # Usually 'Me' isn't a target.
             
             if not my_ignore_stats.empty:
                 cols_to_plot = [c for c in my_ignore_stats.columns if c in ['True Ghost 👻', 'Left on Delivered 📨']]
                 st.bar_chart(my_ignore_stats[cols_to_plot].head(15), horizontal=True) 
             else:
                 st.info("You're a saint! 😇")

        with col_lod2:
             st.write("**People Who Ignore Me** (Them → Me)")
             # Use full_analyzer (need my messages to see if they ignored them)
             them_ignore_stats = full_analyzer_tab6.get_ghosting_stats() # ghosting = them ignoring me
             
             if not them_ignore_stats.empty:
                 # It might return a DataFrame with 'count' or 'True Ghost'/'Left on Delivered' columns?
                 # get_ghosting_stats return structure:
                 # pivot table with 'contact_name' index, columns: 'type' values ('True Ghost', etc)
                 # Wait, let's check analyzer return.
                 # get_ghosting_stats (line 353 analyzer) returns: 
                 # stats = ghost_candidates.groupby(['chat_name', 'type']).size().unstack(fill_value=0)
                 # So columns are 'True Ghost 👻', 'Left on Delivered 📨' etc.
                 
                 cols_to_plot = [c for c in them_ignore_stats.columns if c in ['True Ghost 👻', 'Left on Delivered 📨']]
                 st.bar_chart(them_ignore_stats[cols_to_plot].head(15), horizontal=True)
             else:
                 st.info("Everyone loves you! 💖")
        

            
        st.divider()
        
        # 2. Charts
        c_fun1, c_fun2 = st.columns(2)
        
        with c_fun1:
            st.subheader("🎭 Double Text Ratio")
            st.caption("Percentage of your turns that are double-texts (continuing without reply after >20m).")
            if not beh_scorecard.empty:
                dt_df = beh_scorecard.sort_values('double_text_ratio', ascending=False).head(15)
                fig_dt = px.bar(dt_df, x='double_text_ratio', y=dt_df.index, orientation='h',
                                title="Highest Double Text Ratio",
                                color='gender', color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
                fig_dt.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_title="Double Text %")
                st.plotly_chart(fig_dt, width='stretch')
                
        with c_fun2:
            st.subheader("🌵 Dry Texter Index")
            st.caption("Average words per message.")
            if not fun_stats.empty:
                dry_df = fun_stats.sort_values('avg_word_len', ascending=True).head(15)
                fig_dry = px.bar(dry_df, x='avg_word_len', y=dry_df.index, orientation='h',
                                 title="Shortest Responses (Dryest)",
                                 color='gender', color_discrete_map={'male': '#636EFA', 'female': '#EF553B', 'unknown': 'gray'})
                fig_dry.update_layout(yaxis={'categoryorder':'total descending'}, xaxis_title="Avg Words/Msg")
                st.plotly_chart(fig_dry, width='stretch')
        


    with tab7:
        st.header("🗺️ Location Map")
        loc_data = analyzer.get_location_data()
        if not loc_data.empty:
            st.map(loc_data, latitude='latitude', longitude='longitude')
            st.dataframe(loc_data[['contact_name', 'timestamp', 'place_name']])
        else:
            st.info("No location data found in this backup.")

else:
    st.info("👈 Please enter file paths.")
