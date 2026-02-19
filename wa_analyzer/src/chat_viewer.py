import pandas as pd
import html


def generate_chat_html(
    chat_df,
    flip_sides=False,
    adv_replies=False,
    msgs_above=2,
    msgs_below=1,
    context_view=False,
    collapse_replies=False,
):
    """
    Generates a WhatsApp-like HTML preview of the chat.
    chat_df must be sorted by timestamp.
    """
    if chat_df.empty:
        return "<p>No messages in this chat.</p>"

    # 1. Base Setup
    chat_df = chat_df.copy()

    # Identify sender logic
    # from_me is 1 if it's the device owner.
    # flip_sides inverts it.
    if flip_sides:
        chat_df["is_me"] = chat_df["from_me"] == 0
    else:
        chat_df["is_me"] = chat_df["from_me"] == 1

    # 2. Advanced Highlighting Logic
    highlight_indices = set()
    reply_target_indices = set()
    context_target_indices = set()

    # Create mapping from message_row_id to index for fast lookup
    if "message_row_id" in chat_df.columns:
        row_id_to_idx = {
            row_id: idx for idx, row_id in zip(chat_df.index, chat_df["message_row_id"])
        }
    else:
        row_id_to_idx = {}

    if adv_replies:
        # Find all messages that reply to something
        for idx, row in chat_df.iterrows():
            if pd.notna(row.get("reply_to_row_id")):
                replied_to = row["reply_to_row_id"]
                if replied_to in row_id_to_idx:
                    target_idx = row_id_to_idx[replied_to]
                    reply_target_indices.add(target_idx)

                    # Mark above
                    for i in range(max(0, target_idx - msgs_above), target_idx):
                        highlight_indices.add(i)
                    # Mark below
                    for i in range(
                        target_idx + 1, min(len(chat_df), target_idx + msgs_below + 1)
                    ):
                        highlight_indices.add(i)

    if context_view:
        # If A sends a message to B that doesn't reply to anything, mark the 2 messages just above it
        # (as long as they're from B, ignoring consecutive messages from A).
        for idx in range(1, len(chat_df)):
            row = chat_df.iloc[idx]
            is_reply = pd.notna(row.get("reply_to_row_id"))

            if not is_reply:
                current_sender = row["is_me"]

                # Look above
                prev_idx = idx - 1
                prev_sender = chat_df.iloc[prev_idx]["is_me"]

                if current_sender != prev_sender:
                    # Message from A to B. B is prev_sender.
                    # Mark up to 2 messages above from B, stopping if we hit a message from A.
                    count = 0
                    curr_check = prev_idx
                    while curr_check >= 0 and count < 2:
                        if chat_df.iloc[curr_check]["is_me"] == prev_sender:
                            context_target_indices.add(curr_check)
                            count += 1
                            curr_check -= 1
                        else:
                            break

    # 3. HTML Generation
    css = """
    <style>
        .wa-container {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #e5ddd5;
            background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png");
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            height: 600px;
            overflow-y: auto;
            border-radius: 8px;
            scroll-behavior: smooth;
            position: relative;
        }
        .wa-message-wrap {
            display: flex;
            flex-direction: column;
            width: 100%;
        }
        .wa-message {
            max-width: 65%;
            padding: 8px 12px;
            border-radius: 7.5px;
            position: relative;
            font-size: 14px;
            line-height: 19px;
            box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
            display: inline-block;
            word-wrap: break-word;
            transition: background-color 1.5s;
        }
        .wa-me {
            align-self: flex-end;
            background-color: #dcf8c6;
            border-top-right-radius: 0;
        }
        .wa-them {
            align-self: flex-start;
            background-color: #ffffff;
            border-top-left-radius: 0;
        }
        .wa-time {
            font-size: 11px;
            color: rgba(0,0,0,0.45);
            margin-left: 10px;
            float: right;
            margin-top: 5px;
        }
        .wa-reply {
            background-color: rgba(0,0,0,0.05);
            border-left: 4px solid #35cd96;
            border-radius: 4px;
            padding: 5px;
            margin-bottom: 5px;
            font-size: 13px;
            color: rgba(0,0,0,0.6);
            display: flex;
            flex-direction: column;
            cursor: pointer;
        }
        .wa-reply:hover {
            background-color: rgba(0,0,0,0.1);
        }
        .wa-reply-sender {
            color: #35cd96;
            font-weight: 500;
            margin-bottom: 2px;
        }
        /* Highlights */
        .highlight-reply-target { box-shadow: 0 0 10px 2px #ff9800 !important; border: 2px solid #ff9800; background-color: #fff3e0 !important; }
        .highlight-adv-context { box-shadow: 0 0 10px 2px #03a9f4 !important; border: 2px dashed #03a9f4; opacity: 0.9; }
        .highlight-unreplied-context { box-shadow: 0 0 10px 2px #e91e63 !important; border: 2px solid #e91e63; background-color: #fce4ec !important; }
        .wa-sender-name {
            font-size: 12.5px;
            font-weight: 500;
            color: #35cd96;
            margin-bottom: 2px;
        }
        /* Floating Buttons */
        .wa-floating-btn {
            position: fixed;
            right: 35px;
            background-color: #fff;
            color: #128C7E;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            cursor: pointer;
            font-size: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .wa-floating-btn:hover { background-color: #f0f0f0; }
        #btn-scroll-down { bottom: 40px; }
        #btn-scroll-unreplied { bottom: 90px; }
        /* Collapse block */
        .wa-collapse-block {
            align-self: center;
            background-color: #e2f0fb;
            color: #4a86e8;
            padding: 6px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            margin: 5px 0;
            user-select: none;
        }
        .wa-collapse-block:hover {
            background-color: #c9e2f9;
        }
        .wa-hidden { display: none !important; }
    </style>
    """

    html_parts = [css, '<div class="wa-container" id="wa-chat-container">']

    html_parts.append("""
    <button class="wa-floating-btn" id="btn-scroll-down" onclick="scrollToBottom()" title="Scroll to Bottom">
        <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><polyline points="19 12 12 19 5 12"></polyline></svg>
    </button>
    <button class="wa-floating-btn" id="btn-scroll-unreplied" onclick="scrollToPrevUnrepliedBlock()" title="Go to Previous Unreplied Message">
        <svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
    </button>
    """)

    id_to_text = {}
    id_to_sender = {}
    if "message_row_id" in chat_df.columns:
        for _, row in chat_df.iterrows():
            id_to_text[row["message_row_id"]] = row["text_data"]
            id_to_sender[row["message_row_id"]] = row["is_me"]

    last_sender = None

    collapse_buffer = []
    collapse_group_id = 0

    def render_collapsed_buffer():
        nonlocal collapse_buffer, collapse_group_id
        if not collapse_buffer:
            return ""

        group_html = []
        group_html.append(
            f'<div id="collapse-group-{collapse_group_id}" style="display: flex; flex-direction: column; width: 100%;">'
        )

        total = len(collapse_buffer)

        group_html.append(
            f'<div class="wa-collapse-block" id="collapse-btn-{collapse_group_id}" onclick="expandCollapseGroup({collapse_group_id}, {total})" style="display: flex; align-items: center; justify-content: center; gap: 6px;">'
        )
        expand_amt = min(5, total)
        group_html.append(
            f'<svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg> <span>{total} context messages collapsed. Click to expand {expand_amt}.</span>'
        )
        group_html.append("</div>")

        for i, msg_html in enumerate(collapse_buffer):
            group_html.append(
                f'<div class="wa-hidden collapse-item-{collapse_group_id}" data-index="{i}">{msg_html}</div>'
            )

        group_html.append("</div>")

        collapse_buffer.clear()
        collapse_group_id += 1
        return "\n".join(group_html)

    for idx, row in chat_df.iterrows():
        is_me = row["is_me"]
        text = str(row["text_data"]) if pd.notna(row["text_data"]) else "<Media/Other>"
        # Safe format time
        if pd.notna(row["timestamp"]):
            try:
                time_str = row["timestamp"].strftime("%H:%M")
            except Exception:
                time_str = ""
        else:
            time_str = ""

        classes = ["wa-message"]
        classes.append("wa-me" if is_me else "wa-them")

        if idx in reply_target_indices:
            classes.append("highlight-reply-target")
        if idx in highlight_indices:
            classes.append("highlight-adv-context")
        if idx in context_target_indices:
            classes.append("highlight-unreplied-context")

        class_str = " ".join(classes)

        msg_html_parts = []
        msg_html_parts.append('<div class="wa-message-wrap">')
        msg_html_parts.append(f'<div class="{class_str}" id="msg-{idx}">')

        # Show sender name for 'them' if it's the first message in a sequence
        if not is_me and last_sender != is_me:
            sender_name = "Them"
            if "contact_name" in row and pd.notna(row["contact_name"]):
                sender_name = str(row["contact_name"])
            elif "raw_string" in row and pd.notna(row["raw_string"]):
                sender_name = str(row["raw_string"])
            msg_html_parts.append(
                f'<div class="wa-sender-name">{html.escape(sender_name)}</div>'
            )

        # Reply block
        reply_to_id = row.get("reply_to_row_id")
        if pd.notna(reply_to_id) and reply_to_id in id_to_text:
            rep_text = id_to_text[reply_to_id]
            rep_sender = "You" if id_to_sender.get(reply_to_id, False) else "Them"
            safe_text = str(rep_text) if pd.notna(rep_text) else "<Media>"

            target_idx = row_id_to_idx.get(reply_to_id, -1)
            scroll_onclick = (
                f"scrollToMsg('msg-{target_idx}')" if target_idx != -1 else ""
            )

            msg_html_parts.append(f"""
            <div class="wa-reply" onclick="{scroll_onclick}">
                <span class="wa-reply-sender">{rep_sender}</span>
                <span class="wa-reply-text">{html.escape(safe_text[:100])}{"..." if len(safe_text) > 100 else ""}</span>
            </div>
            """)

        msg_html_parts.append(
            f"<span>{html.escape(text).replace(chr(10), '<br>')}</span>"
        )
        msg_html_parts.append(f'<span class="wa-time">{time_str}</span>')
        msg_html_parts.append("</div>")
        msg_html_parts.append("</div>")

        msg_html = "\n".join(msg_html_parts)

        if collapse_replies and (
            idx in reply_target_indices or idx in highlight_indices
        ):
            collapse_buffer.append(msg_html)
        else:
            if collapse_buffer:
                html_parts.append(render_collapsed_buffer())
            html_parts.append(msg_html)

        last_sender = is_me

    if collapse_buffer:
        html_parts.append(render_collapsed_buffer())

    script = """
    <script>
        var container = document.getElementById("wa-chat-container");
        if(container) {
            container.scrollTop = container.scrollHeight;
        }
        
        function scrollToBottom() {
            var cont = document.getElementById("wa-chat-container");
            cont.scrollTo({ top: cont.scrollHeight, behavior: 'smooth' });
        }
        
        let unrepliedTargets = [];
        let currentTargetIdx = -1;

        function buildUnrepliedTargets() {
            let messages = document.querySelectorAll('.wa-message');
            unrepliedTargets = [];
            let inUnrepliedBlock = false;
            let lastUnrepliedMsg = null;

            for (let i = 0; i < messages.length; i++) {
                let msg = messages[i];
                let isHighlighted = msg.classList.contains('highlight-reply-target') ||
                                    msg.classList.contains('highlight-adv-context') ||
                                    msg.classList.contains('highlight-unreplied-context');

                if (!isHighlighted) {
                    lastUnrepliedMsg = msg;
                    inUnrepliedBlock = true;
                } else {
                    if (inUnrepliedBlock && lastUnrepliedMsg) {
                        unrepliedTargets.push(lastUnrepliedMsg);
                        inUnrepliedBlock = false;
                    }
                }
            }
            if (inUnrepliedBlock && lastUnrepliedMsg) {
                unrepliedTargets.push(lastUnrepliedMsg);
            }
            currentTargetIdx = unrepliedTargets.length - 1;
        }

        function scrollToPrevUnrepliedBlock() {
            if (unrepliedTargets.length === 0) {
                buildUnrepliedTargets();
            }
            if (unrepliedTargets.length === 0) {
                return;
            }

            if (currentTargetIdx < 0) {
                currentTargetIdx = unrepliedTargets.length - 1;
            }

            let target = unrepliedTargets[currentTargetIdx];

            var hiddenWrap = target.closest('.wa-hidden');
            if (hiddenWrap) {
                var classList = hiddenWrap.className.split(' ');
                var groupClass = classList.find(c => c.startsWith('collapse-item-'));
                if (groupClass) {
                    var groupId = groupClass.split('-')[2];
                    expandCollapseGroupAll(groupId);
                }
            }

            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            var oldBg = target.style.backgroundColor;
            target.style.backgroundColor = '#f8bbd0';
            setTimeout(() => { target.style.backgroundColor = oldBg; }, 1500);

            currentTargetIdx--;
        }
        
        function scrollToMsg(msgId) {
            var msg = document.getElementById(msgId);
            if (msg) {
                var hiddenWrap = msg.closest('.wa-hidden');
                if (hiddenWrap) {
                    var classList = hiddenWrap.className.split(' ');
                    var groupClass = classList.find(c => c.startsWith('collapse-item-'));
                    if (groupClass) {
                        var groupId = groupClass.split('-')[2];
                        expandCollapseGroupAll(groupId);
                    }
                }
                
                msg.scrollIntoView({ behavior: 'smooth', block: 'center' });
                var oldBg = msg.style.backgroundColor;
                msg.style.backgroundColor = '#ffe0b2';
                setTimeout(() => { msg.style.backgroundColor = oldBg; }, 1500);
            }
        }
        
        var groupShownCount = {};
        
        function expandCollapseGroup(groupId, totalCount) {
            if (!groupShownCount[groupId]) {
                groupShownCount[groupId] = 0;
            }
            
            var currentShown = groupShownCount[groupId];
            var toShow = Math.min(currentShown + 5, totalCount);
            
            var items = document.getElementsByClassName('collapse-item-' + groupId);
            for (var i = currentShown; i < toShow; i++) {
                if (items[i]) {
                    items[i].classList.remove('wa-hidden');
                }
            }
            
            groupShownCount[groupId] = toShow;
            var btn = document.getElementById('collapse-btn-' + groupId);
            
            if (toShow >= totalCount) {
                btn.style.display = 'none';
            } else {
                var remaining = totalCount - toShow;
                var nextExpand = Math.min(5, remaining);
                btn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg> <span>' + remaining + ' more hidden. Click to expand ' + nextExpand + '.</span>';
            }
        }
        
        function expandCollapseGroupAll(groupId) {
            var items = document.getElementsByClassName('collapse-item-' + groupId);
            for (var i = 0; i < items.length; i++) {
                items[i].classList.remove('wa-hidden');
            }
            var btn = document.getElementById('collapse-btn-' + groupId);
            if (btn) btn.style.display = 'none';
        }
    </script>
    """
    html_parts.append(script)
    html_parts.append("</div>")

    return "\n".join(html_parts)


def export_chat_json(chat_df):
    export_df = chat_df.copy()
    if "timestamp" in export_df.columns:
        export_df["timestamp"] = export_df["timestamp"].astype(str)
    return export_df.to_json(orient="records", date_format="iso")


def export_chat_txt(chat_df, flip_sides=False):
    lines = []

    id_to_text = {}
    if "message_row_id" in chat_df.columns:
        for _, row in chat_df.iterrows():
            id_to_text[row["message_row_id"]] = row["text_data"]

    for _, row in chat_df.iterrows():
        is_me = row["from_me"] == 1
        if flip_sides:
            is_me = not is_me

        sender = "Me" if is_me else str(row.get("contact_name", "Them"))
        time_str = ""
        if pd.notna(row["timestamp"]):
            try:
                time_str = row["timestamp"].strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass

        text = str(row["text_data"]) if pd.notna(row["text_data"]) else "<Media/Other>"

        reply_to_id = row.get("reply_to_row_id")
        reply_str = ""
        if pd.notna(reply_to_id) and reply_to_id in id_to_text:
            rep_text = str(id_to_text[reply_to_id]).replace(chr(10), " ")[:50]
            reply_str = f" [Replying to: {rep_text}...]"

        text = text.replace(chr(10), " | ")
        lines.append(f"[{time_str}] {sender}: {text}{reply_str}")

    return "\n".join(lines)
