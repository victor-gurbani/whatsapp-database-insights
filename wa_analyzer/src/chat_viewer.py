import pandas as pd
import html
import json


def generate_chat_html(
    chat_df,
    flip_sides=False,
    adv_replies=False,
    msgs_above=2,
    msgs_below=1,
    context_view=False,
    unreplied_chunk_size=1,
    unreplied_other_only=False,
    collapse_replies=False,
    virtualization=True,
    virt_initial=300,
    virt_chunk=120,
    my_name="Me",
):
    if chat_df.empty:
        return "<p>No messages in this chat.</p>"

    # 1. Base Setup
    chat_df = chat_df.copy()

    if flip_sides:
        chat_df["is_me"] = chat_df["from_me"] == 0
    else:
        chat_df["is_me"] = chat_df["from_me"] == 1

    # 2. Advanced Highlighting Logic
    highlight_indices = set()
    reply_target_indices = set()
    context_target_indices = set()

    if "message_row_id" in chat_df.columns:
        row_id_to_idx = {
            row_id: idx for idx, row_id in zip(chat_df.index, chat_df["message_row_id"])
        }
    else:
        row_id_to_idx = {}

    if adv_replies:
        for idx, row in chat_df.iterrows():
            if pd.notna(row.get("reply_to_row_id")):
                replied_to = row["reply_to_row_id"]
                if replied_to in row_id_to_idx:
                    target_idx = row_id_to_idx[replied_to]
                    target_is_me = chat_df.iloc[target_idx]["is_me"]

                    if not target_is_me:
                        reply_target_indices.add(target_idx)
                        for i in range(max(0, target_idx - msgs_above), target_idx):
                            if not chat_df.iloc[i]["is_me"]:
                                highlight_indices.add(i)
                        for i in range(
                            target_idx + 1,
                            min(len(chat_df), target_idx + msgs_below + 1),
                        ):
                            if not chat_df.iloc[i]["is_me"]:
                                highlight_indices.add(i)

    if context_view:
        for idx in range(1, len(chat_df)):
            row = chat_df.iloc[idx]
            is_reply = pd.notna(row.get("reply_to_row_id"))

            if not is_reply:
                current_sender = row["is_me"]
                prev_idx = idx - 1
                prev_sender = chat_df.iloc[prev_idx]["is_me"]

                if current_sender != prev_sender:
                    count = 0
                    curr_check = prev_idx
                    while curr_check >= 0 and count < 2:
                        if chat_df.iloc[curr_check]["is_me"] == prev_sender:
                            context_target_indices.add(curr_check)
                            count += 1
                            curr_check -= 1
                        else:
                            break

    # 3. Serialize data to JSON for the frontend
    messages_data = []

    id_to_text = {}
    id_to_sender = {}
    id_to_from_me = {}
    if "message_row_id" in chat_df.columns:
        for _, row in chat_df.iterrows():
            id_to_text[row["message_row_id"]] = row["text_data"]
            id_to_sender[row["message_row_id"]] = row["is_me"]
            id_to_from_me[row["message_row_id"]] = row["from_me"]

    last_sender = None
    for idx, row in chat_df.iterrows():
        is_me = bool(row["is_me"])
        text = str(row["text_data"]) if pd.notna(row["text_data"]) else ""
        mime_type = (
            str(row["mime_type"])
            if "mime_type" in row and pd.notna(row["mime_type"])
            else ""
        )

        if pd.notna(row.get("timestamp")):
            try:
                time_str = row["timestamp"].strftime("%H:%M")
            except Exception:
                time_str = ""
        else:
            time_str = ""

        highlight_class = ""
        if idx in reply_target_indices:
            highlight_class = "highlight-reply-target"
        elif idx in highlight_indices:
            highlight_class = "highlight-adv-context"
        elif idx in context_target_indices:
            highlight_class = "highlight-unreplied-context"

        sender_name = ""
        if not is_me and last_sender != is_me:
            if flip_sides and row.get("from_me") == 1:
                sender_name = str(my_name)
            else:
                sender_name = "Them"
                if "contact_name" in row and pd.notna(row["contact_name"]):
                    sender_name = str(row["contact_name"])
                elif "raw_string" in row and pd.notna(row["raw_string"]):
                    sender_name = str(row["raw_string"])

        reply_to_id = row.get("reply_to_row_id")
        rep_text = ""
        rep_sender = ""
        rep_target_idx = -1
        if pd.notna(reply_to_id) and reply_to_id in id_to_text:
            rep_text = (
                str(id_to_text[reply_to_id])
                if pd.notna(id_to_text[reply_to_id])
                else "<Media>"
            )
            rep_is_me = id_to_sender.get(reply_to_id, False)
            if rep_is_me:
                rep_sender = "You"
            else:
                if flip_sides and id_to_from_me.get(reply_to_id) == 1:
                    rep_sender = str(my_name)
                else:
                    rep_sender = "Them"
            rep_target_idx = row_id_to_idx.get(reply_to_id, -1)

        should_collapse = collapse_replies and (
            idx in reply_target_indices or idx in highlight_indices
        )

        messages_data.append(
            {
                "idx": idx,
                "is_me": is_me,
                "text": text,
                "mime_type": mime_type,
                "time_str": time_str,
                "highlight_class": highlight_class,
                "sender_name": sender_name,
                "rep_text": rep_text,
                "rep_sender": rep_sender,
                "rep_target_idx": rep_target_idx,
                "should_collapse": bool(should_collapse),
            }
        )

        last_sender = is_me

    chat_json_str = json.dumps(messages_data)
    virt_str = "true" if virtualization else "false"

    # 4. HTML + CSS + JS Template
    css = """
    <style>
        .wa-wrapper-top { display: flex; flex-direction: column; gap: 10px; height: 600px; overflow: hidden; }
        .wa-search-bar { flex-shrink: 0; padding: 10px; font-size: 14px; border: 1px solid #ccc; border-radius: 8px; width: 100%; box-sizing: border-box; box-shadow: 0 1px 3px rgba(0,0,0,0.1); outline: none; }
        .wa-search-bar:focus { border-color: #35cd96; }
        .wa-container { flex: 1; min-height: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #e5ddd5; background-image: url("https://user-images.githubusercontent.com/15075759/28719144-86dc0f70-73b1-11e7-911d-60d70fcded21.png"); padding: 20px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; border-radius: 8px; scroll-behavior: smooth; position: relative; }
        .wa-message-wrap { display: flex; flex-direction: column; width: 100%; }
        .wa-message { max-width: 65%; padding: 8px 12px; border-radius: 7.5px; position: relative; font-size: 14px; line-height: 19px; box-shadow: 0 1px 0.5px rgba(0,0,0,0.13); display: inline-block; word-wrap: break-word; transition: background-color 1.5s; }
        .wa-me { align-self: flex-end; background-color: #dcf8c6; border-top-right-radius: 0; }
        .wa-them { align-self: flex-start; background-color: #ffffff; border-top-left-radius: 0; }
        .wa-time { font-size: 11px; color: rgba(0,0,0,0.45); margin-left: 10px; float: right; margin-top: 5px; }
        .wa-reply { background-color: rgba(0,0,0,0.05); border-left: 4px solid #35cd96; border-radius: 4px; padding: 5px; margin-bottom: 5px; font-size: 13px; color: rgba(0,0,0,0.6); display: flex; flex-direction: column; cursor: pointer; }
        .wa-reply:hover { background-color: rgba(0,0,0,0.1); }
        .wa-reply-sender { color: #35cd96; font-weight: 500; margin-bottom: 2px; }
        .highlight-reply-target { box-shadow: 0 0 10px 2px #ff9800 !important; border: 2px solid #ff9800; background-color: #fff3e0 !important; }
        .highlight-adv-context { box-shadow: 0 0 10px 2px #03a9f4 !important; border: 2px dashed #03a9f4; opacity: 0.9; }
        .highlight-unreplied-context { box-shadow: 0 0 10px 2px #e91e63 !important; border: 2px solid #e91e63; background-color: #fce4ec !important; }
        .highlight-search { background-color: #fff59d !important; color: #000; padding: 0 2px; border-radius: 3px; }
        .wa-sender-name { font-size: 12.5px; font-weight: 500; color: #35cd96; margin-bottom: 2px; }
        .wa-floating-btn { position: fixed; right: 35px; background-color: #fff; color: #128C7E; border: none; border-radius: 50%; width: 40px; height: 40px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); cursor: pointer; font-size: 20px; display: flex; align-items: center; justify-content: center; z-index: 1000; }
        .wa-floating-btn:hover { background-color: #f0f0f0; }
        #btn-scroll-down { bottom: 40px; }
        #btn-scroll-unreplied { bottom: 90px; }
        #btn-scroll-unreplied-q { bottom: 140px; }
        .wa-collapse-block { align-self: center; background-color: #e2f0fb; color: #4a86e8; padding: 6px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; cursor: pointer; box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin: 5px 0; user-select: none; }
        .wa-collapse-block:hover { background-color: #c9e2f9; }
        .wa-hidden { display: none !important; }
        .wa-media-placeholder { display: flex; align-items: center; gap: 8px; background-color: rgba(0,0,0,0.05); padding: 10px; border-radius: 6px; margin-bottom: 5px; color: #555; font-weight: 500; }
        .virtual-spacer { width: 100%; display: flex; justify-content: center; padding: 10px 0; }
        .load-more-btn { background-color: #e2f0fb; color: #4a86e8; border: none; padding: 8px 16px; border-radius: 16px; cursor: pointer; font-size: 13px; font-weight: bold; box-shadow: 0 1px 2px rgba(0,0,0,0.1); }
        .load-more-btn:hover { background-color: #c9e2f9; }
    </style>
    """

    html_parts = [
        css,
        '<div class="wa-wrapper-top">',
        '<input type="text" id="wa-search-input" class="wa-search-bar" placeholder="🔍 Search in chat... (Press Enter to highlight)">',
        '<div class="wa-container" id="wa-chat-container">',
        '<div id="wa-virtual-top-spacer" class="virtual-spacer" style="display:none;"><button class="load-more-btn" onclick="loadMoreUpwards()">Load older messages</button></div>',
        '<div id="wa-render-anchor"></div>',
        '<button class="wa-floating-btn" id="btn-scroll-down" onclick="scrollToBottom()" title="Scroll to Bottom"><svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><polyline points="19 12 12 19 5 12"></polyline></svg></button>',
        '<button class="wa-floating-btn" id="btn-scroll-unreplied" onclick="scrollToPrevUnrepliedBlock()" title="Go to Previous Unreplied Message"><svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg></button>',
        '<button class="wa-floating-btn" id="btn-scroll-unreplied-q" onclick="scrollToPrevUnrepliedQuestion()" title="Go to Previous Unreplied Question"><svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path><line x1="12" y1="17" x2="12.01" y2="17"></line></svg></button>',
        "</div></div>",
    ]

    script = """
    <script>
        const CHAT_DATA = __CHAT_DATA_JSON__;
        const VIRTUALIZATION_ENABLED = __VIRT_FLAG__;
        const CHUNK_SIZE = __VIRT_CHUNK__;
        const INITIAL_CHUNK_SIZE = __VIRT_INITIAL__;
        const UNREPLIED_CHUNK_SIZE = __UNREPLIED_CHUNK_SIZE__;
        const UNREPLIED_OTHER_ONLY = __UNREPLIED_OTHER_ONLY__;
        
        let renderedStartIndex = 0;
        let renderedEndIndex = 0;
        let collapseBuffer = [];
        let collapseGroupId = 0;
        let isLoading = false;
        
        const container = document.getElementById("wa-chat-container");
        const renderAnchor = document.getElementById("wa-render-anchor");
        const topSpacer = document.getElementById("wa-virtual-top-spacer");
        const searchInput = document.getElementById("wa-search-input");
        let currentSearchQuery = "";
        
        const icons = {
            image: '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>',
            video: '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>',
            audio: '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>',
            document: '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>',
            default: '<svg viewBox="0 0 24 24" width="20" height="20" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
        };
        
        function escapeHtml(unsafe) {
            return String(unsafe).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }
        
        function highlightText(text, query) {
            if (!query) return escapeHtml(text).replace(/\\n/g, '<br>');
            const safeQuery = escapeHtml(query).replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
            const regex = new RegExp(`(${safeQuery})`, 'gi');
            return escapeHtml(text).replace(regex, '<span class="highlight-search">$1</span>').replace(/\\n/g, '<br>');
        }
        
        function getMediaHtml(mimeType) {
            if (!mimeType) return "";
            mimeType = mimeType.toLowerCase();
            let icon = icons.default; let label = "Media";
            if (mimeType.includes("image")) { icon = icons.image; label = "Photo"; }
            else if (mimeType.includes("video")) { icon = icons.video; label = "Video"; }
            else if (mimeType.includes("audio")) { icon = icons.audio; label = "Audio"; }
            else if (mimeType.includes("pdf") || mimeType.includes("document")) { icon = icons.document; label = "Document"; }
            return `<div class="wa-media-placeholder">${icon} <span>${label}</span></div>`;
        }

        function buildMessageHtml(msg) {
            let htmlParts = [];
            let classStr = "wa-message " + (msg.is_me ? "wa-me" : "wa-them");
            if (msg.highlight_class) classStr += " " + msg.highlight_class;
            
            htmlParts.push('<div class="wa-message-wrap">');
            htmlParts.push(`<div class="${classStr}" id="msg-${msg.idx}">`);
            if (msg.sender_name) htmlParts.push(`<div class="wa-sender-name">${escapeHtml(msg.sender_name)}</div>`);
            
            if (msg.rep_text || msg.rep_sender) {
                let safeText = msg.rep_text ? msg.rep_text : "Media";
                let scrollClick = msg.rep_target_idx !== -1 ? `scrollToMsg('msg-${msg.rep_target_idx}')` : '';
                let textTrunc = safeText.length > 100 ? escapeHtml(safeText.substring(0,100)) + '...' : escapeHtml(safeText);
                htmlParts.push(`
                <div class="wa-reply" onclick="${scrollClick}">
                    <span class="wa-reply-sender">${escapeHtml(msg.rep_sender)}</span>
                    <span class="wa-reply-text">${textTrunc}</span>
                </div>
                `);
            }
            
            let contentHtml = "";
            if (msg.text) contentHtml += `<span>${highlightText(msg.text, currentSearchQuery)}</span>`;
            else if (msg.mime_type) contentHtml += getMediaHtml(msg.mime_type);
            else contentHtml += `<span>&lt;Media/Other&gt;</span>`;
            
            htmlParts.push(contentHtml);
            htmlParts.push(`<span class="wa-time">${msg.time_str}</span>`);
            htmlParts.push('</div></div>');
            return htmlParts.join("\\n");
        }

        function renderCollapsedBuffer() {
            if (collapseBuffer.length === 0) return "";
            let total = collapseBuffer.length;
            let expandAmt = Math.min(5, total);
            let html = [];
            html.push(`<div id="collapse-group-${collapseGroupId}" style="display: flex; flex-direction: column; width: 100%;">`);
            html.push(`
                <div class="wa-collapse-block" id="collapse-btn-${collapseGroupId}" onclick="expandCollapseGroup(${collapseGroupId}, ${total})" style="display: flex; align-items: center; justify-content: center; gap: 6px;">
                    <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg> 
                    <span>${total} context messages collapsed. Click to expand ${expandAmt}.</span>
                </div>
            `);
            for (let i = 0; i < collapseBuffer.length; i++) {
                html.push(`<div class="wa-hidden collapse-item-${collapseGroupId}" data-index="${i}">${collapseBuffer[i]}</div>`);
            }
            html.push('</div>');
            collapseBuffer = [];
            collapseGroupId++;
            return html.join("\\n");
        }

        function renderMessages(startIndex, endIndex, prepend = false) {
            let htmlBuffer = [];
            for (let i = startIndex; i < endIndex; i++) {
                let msg = CHAT_DATA[i];
                let msgHtml = buildMessageHtml(msg);
                
                // If there's a search query, DO NOT collapse any messages so the user can easily see search results.
                if (msg.should_collapse && !currentSearchQuery) {
                    collapseBuffer.push(msgHtml);
                } else {
                    if (collapseBuffer.length > 0) htmlBuffer.push(renderCollapsedBuffer());
                    htmlBuffer.push(msgHtml);
                }
            }
            if (collapseBuffer.length > 0) htmlBuffer.push(renderCollapsedBuffer());
            
            let wrapper = document.createElement('div');
            wrapper.innerHTML = htmlBuffer.join("\\n");
            
            if (prepend) {
                let oldScrollHeight = container.scrollHeight;
                let oldScrollTop = container.scrollTop;
                
                container.style.overflowAnchor = "none";
                renderAnchor.insertBefore(wrapper, renderAnchor.firstChild);
                
                let newScrollHeight = container.scrollHeight;
                container.scrollTop = oldScrollTop + (newScrollHeight - oldScrollHeight);
                container.style.overflowAnchor = "auto";
            } else {
                renderAnchor.appendChild(wrapper);
            }
            
            renderedStartIndex = Math.min(renderedStartIndex, startIndex);
            renderedEndIndex = Math.max(renderedEndIndex, endIndex);
            topSpacer.style.display = renderedStartIndex > 0 ? "flex" : "none";
        }

        function loadMoreUpwards() {
            if (renderedStartIndex <= 0 || isLoading) return;
            isLoading = true;
            let btn = topSpacer.querySelector('.load-more-btn');
            if (btn) btn.innerHTML = "Loading...";
            
            // tiny timeout to let UI update
            setTimeout(() => {
                let newStart = Math.max(0, renderedStartIndex - CHUNK_SIZE);
                renderMessages(newStart, renderedStartIndex, true);
                if (btn) btn.innerHTML = "Load older messages";
                isLoading = false;
            }, 10);
        }

        function scrollToBottomInstant() {
            container.scrollTop = container.scrollHeight;
            requestAnimationFrame(() => {
                container.scrollTop = container.scrollHeight;
            });
            setTimeout(() => {
                container.scrollTop = container.scrollHeight;
            }, 50);
            setTimeout(() => {
                container.scrollTop = container.scrollHeight;
            }, 150);
        }

        function initChat() {
            targetsBuilt = false;
            unrepliedTargets = [];
            unrepliedQuestionTargets = [];
            renderAnchor.innerHTML = "";
            collapseBuffer = [];
            collapseGroupId = 0;
            let total = CHAT_DATA.length;
            if (total === 0) return;
            
            if (VIRTUALIZATION_ENABLED && !currentSearchQuery) {
                renderedStartIndex = Math.max(0, total - INITIAL_CHUNK_SIZE);
                renderedEndIndex = renderedStartIndex;
                renderMessages(renderedStartIndex, total, false);
                scrollToBottomInstant();
            } else {
                renderedStartIndex = 0;
                renderedEndIndex = 0;
                renderMessages(0, total, false);
                scrollToBottomInstant();
            }
            
            if (VIRTUALIZATION_ENABLED && !currentSearchQuery && window.IntersectionObserver) {
                let observer = new IntersectionObserver((entries) => {
                    if (entries[0].isIntersecting && renderedStartIndex > 0 && !isLoading) {
                        loadMoreUpwards();
                    }
                }, { root: container, rootMargin: "150px 0px 0px 0px" });
                observer.observe(topSpacer);
            }
        }

        searchInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                currentSearchQuery = searchInput.value.trim();
                initChat();
                if (currentSearchQuery) {
                    setTimeout(() => {
                        let firstMatch = document.querySelector('.highlight-search');
                        if (firstMatch) firstMatch.scrollIntoView({behavior: 'smooth', block: 'center'});
                    }, 100);
                }
            }
        });
        
        function scrollToBottom() {
            container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
        }
        
        let unrepliedTargets = [];
        let currentTargetIdx = -1;
        let unrepliedQuestionTargets = [];
        let currentQuestionTargetIdx = -1;
        let targetsBuilt = false;

        function buildAllUnrepliedTargets() {
            let messages = document.querySelectorAll('.wa-message');
            unrepliedTargets = [];
            unrepliedQuestionTargets = [];
            let currentBlock = [];

            function processBlock() {
                if (currentBlock.length >= UNREPLIED_CHUNK_SIZE) {
                    // Normal Target: Last message in block matching criteria
                    let target = null;
                    for (let j = currentBlock.length - 1; j >= 0; j--) {
                        let cMsg = currentBlock[j];
                        if (!UNREPLIED_OTHER_ONLY || !cMsg.classList.contains('wa-me')) {
                            target = cMsg;
                            break;
                        }
                    }
                    if (target) {
                        unrepliedTargets.push(target);
                    }

                    // Question Targets: All messages in block matching criteria that have '?'
                    for (let j = 0; j < currentBlock.length; j++) {
                        let cMsg = currentBlock[j];
                        if (!UNREPLIED_OTHER_ONLY || !cMsg.classList.contains('wa-me')) {
                            if (cMsg.textContent && cMsg.textContent.includes('?')) {
                                unrepliedQuestionTargets.push(cMsg);
                            }
                        }
                    }
                }
                currentBlock = [];
            }

            for (let i = 0; i < messages.length; i++) {
                let msg = messages[i];
                let isHighlighted = msg.classList.contains('highlight-reply-target') ||
                                    msg.classList.contains('highlight-adv-context') ||
                                    msg.classList.contains('highlight-unreplied-context');

                if (!isHighlighted) {
                    currentBlock.push(msg);
                } else {
                    processBlock();
                }
            }
            processBlock();

            currentTargetIdx = unrepliedTargets.length - 1;
            currentQuestionTargetIdx = unrepliedQuestionTargets.length - 1;
            targetsBuilt = true;
        }

        function scrollToPrevUnrepliedQuestion() {
            if (!targetsBuilt) buildAllUnrepliedTargets();
            if (unrepliedQuestionTargets.length === 0) return;

            if (currentQuestionTargetIdx < 0) currentQuestionTargetIdx = unrepliedQuestionTargets.length - 1;
            let target = unrepliedQuestionTargets[currentQuestionTargetIdx];

            var hiddenWrap = target.closest('.wa-hidden');
            if (hiddenWrap) {
                var classList = hiddenWrap.className.split(' ');
                var groupClass = classList.find(c => c.startsWith('collapse-item-'));
                if (groupClass) expandCollapseGroupAll(groupClass.split('-')[2]);
            }

            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            var oldBg = target.style.backgroundColor;
            target.style.backgroundColor = '#ffeb3b';
            setTimeout(() => { target.style.backgroundColor = oldBg; }, 1500);
            currentQuestionTargetIdx--;
        }

        function scrollToPrevUnrepliedBlock() {
            if (!targetsBuilt) buildAllUnrepliedTargets();
            if (unrepliedTargets.length === 0) return;

            if (currentTargetIdx < 0) currentTargetIdx = unrepliedTargets.length - 1;
            let target = unrepliedTargets[currentTargetIdx];

            var hiddenWrap = target.closest('.wa-hidden');
            if (hiddenWrap) {
                var classList = hiddenWrap.className.split(' ');
                var groupClass = classList.find(c => c.startsWith('collapse-item-'));
                if (groupClass) expandCollapseGroupAll(groupClass.split('-')[2]);
            }

            target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            var oldBg = target.style.backgroundColor;
            target.style.backgroundColor = '#f8bbd0';
            setTimeout(() => { target.style.backgroundColor = oldBg; }, 1500);
            currentTargetIdx--;
        }
        
        function scrollToMsg(msgId) {
            var msg = document.getElementById(msgId);
            if (!msg && VIRTUALIZATION_ENABLED) {
                let wasSearch = currentSearchQuery;
                currentSearchQuery = "force_load_all";
                initChat();
                currentSearchQuery = wasSearch;
                msg = document.getElementById(msgId);
            }
            
            if (msg) {
                var hiddenWrap = msg.closest('.wa-hidden');
                if (hiddenWrap) {
                    var classList = hiddenWrap.className.split(' ');
                    var groupClass = classList.find(c => c.startsWith('collapse-item-'));
                    if (groupClass) expandCollapseGroupAll(groupClass.split('-')[2]);
                }
                
                msg.scrollIntoView({ behavior: 'smooth', block: 'center' });
                var oldBg = msg.style.backgroundColor;
                msg.style.backgroundColor = '#ffe0b2';
                setTimeout(() => { msg.style.backgroundColor = oldBg; }, 1500);
            }
        }
        
        var groupShownCount = {};
        
        function expandCollapseGroup(groupId, totalCount) {
            if (!groupShownCount[groupId]) groupShownCount[groupId] = 0;
            var currentShown = groupShownCount[groupId];
            var toShow = Math.min(currentShown + 5, totalCount);
            
            var items = document.getElementsByClassName('collapse-item-' + groupId);
            for (var i = currentShown; i < toShow; i++) {
                if (items[i]) items[i].classList.remove('wa-hidden');
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
            for (var i = 0; i < items.length; i++) items[i].classList.remove('wa-hidden');
            var btn = document.getElementById('collapse-btn-' + groupId);
            if (btn) btn.style.display = 'none';
        }

        initChat();
    </script>
    """

    script = script.replace("__CHAT_DATA_JSON__", chat_json_str)
    script = script.replace("__VIRT_FLAG__", virt_str)
    script = script.replace("__VIRT_INITIAL__", str(virt_initial))
    script = script.replace("__VIRT_CHUNK__", str(virt_chunk))
    script = script.replace("__UNREPLIED_CHUNK_SIZE__", str(unreplied_chunk_size))
    script = script.replace(
        "__UNREPLIED_OTHER_ONLY__", "true" if unreplied_other_only else "false"
    )

    html_parts.append(script)

    return "\n".join(html_parts)


def export_chat_html_standalone(chat_df, **kwargs):
    raw_html = generate_chat_html(chat_df, **kwargs)

    raw_html = raw_html.replace(
        "height: 600px;",
        "height: 100vh; width: 100vw; box-sizing: border-box; border-radius: 0; margin: 0;",
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WhatsApp Chat Export</title>
    <style>
        body {{ margin: 0; padding: 0; background-color: #e5ddd5; overflow: hidden; }}
    </style>
</head>
<body>
    {raw_html}
</body>
</html>
"""
    return full_html


def export_chat_json(chat_df):
    export_df = chat_df.copy()
    if "timestamp" in export_df.columns:
        export_df["timestamp"] = export_df["timestamp"].astype(str)
    return export_df.to_json(orient="records", date_format="iso")


def export_chat_txt(chat_df, flip_sides=False, my_name="Me"):
    lines = []

    id_to_text = {}
    if "message_row_id" in chat_df.columns:
        for _, row in chat_df.iterrows():
            id_to_text[row["message_row_id"]] = row["text_data"]

    for _, row in chat_df.iterrows():
        is_me = row["from_me"] == 1
        if flip_sides:
            is_me = not is_me

        if is_me:
            sender = "Me"
        else:
            if flip_sides and row.get("from_me") == 1:
                sender = str(my_name)
            else:
                sender = str(row.get("contact_name", "Them"))
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
