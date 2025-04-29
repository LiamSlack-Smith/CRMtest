// static/js/chat.js
// Handles chat interactions, message display, SSE, and file viewer logic.

// --- DOM Elements (Chat & Viewer Specific) ---
const chatbox = document.getElementById('chatbox');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const fileViewer = document.getElementById('file-viewer');
const fileViewerTitle = document.getElementById('file-viewer-title');
const fileViewerContent = document.getElementById('file-viewer-content');
const closeViewerBtn = document.getElementById('close-viewer-btn');
const viewerResizer = document.getElementById('viewer-resizer');

// --- State Variables (Chat & Viewer Specific) ---
let currentAssistantMessageElement = null;
let accumulatedResponseForMarkdown = '';
let sourcesForCurrentMessage = [];
let markdownUpdateTimeoutId = null;
let isStreaming = false;
const MARKDOWN_UPDATE_INTERVAL_MS = 80; // Interval for updating markdown during streaming
let isResizingViewer = false;
let viewerStartX = 0;
let viewerStartWidth = 0;

// --- Chat Helper Functions ---
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(sender === 'user' ? 'user-message' : 'assistant-message');

    if (sender === 'user') {
        // Sanitize user input before inserting as HTML
        const tempDiv = document.createElement('div');
        tempDiv.textContent = text;
        messageDiv.innerHTML = tempDiv.innerHTML.replace(/\n/g, '<br>');
    } else {
        // Assistant message content is handled by scheduleMarkdownUpdate
        const contentDiv = document.createElement('div');
        contentDiv.classList.add('message-content');
        contentDiv.innerHTML = text || '<span class="blinking-cursor">▍</span>'; // Initial state
        messageDiv.appendChild(contentDiv);
    }
    if (chatbox) {
        chatbox.appendChild(messageDiv);
        scrollToBottom();
    } else {
        console.error("Chatbox element not found!");
    }
    return messageDiv;
}

function scrollToBottom() {
    if (!chatbox) return;
    const isScrolledToBottom = chatbox.scrollHeight - chatbox.clientHeight <= chatbox.scrollTop + 60; // Threshold
    if (isScrolledToBottom) {
        if ('scrollBehavior' in document.documentElement.style) {
            chatbox.scrollTo({ top: chatbox.scrollHeight, behavior: 'smooth' });
        } else {
            chatbox.scrollTop = chatbox.scrollHeight; // Fallback for older browsers
        }
    }
}

function scheduleMarkdownUpdate(force = false) {
    if (markdownUpdateTimeoutId && !force) return; // Already scheduled
    if (markdownUpdateTimeoutId) clearTimeout(markdownUpdateTimeoutId);

    const performUpdate = () => {
        if (!currentAssistantMessageElement) { markdownUpdateTimeoutId = null; return; }
        const contentElement = currentAssistantMessageElement.querySelector('.message-content');
        if (!contentElement) { console.error("No message-content div."); markdownUpdateTimeoutId = null; return; }

        try {
            let htmlContent = marked.parse(accumulatedResponseForMarkdown);
            if (isStreaming && !force) { htmlContent += '<span class="blinking-cursor">▍</span>'; }
            contentElement.innerHTML = htmlContent;
            if (force) { appendSources(currentAssistantMessageElement, sourcesForCurrentMessage); } // Append sources on final update
            scrollToBottom();
        } catch (e) {
            console.error("Markdown update error:", e);
            contentElement.innerHTML += '<br><span class="error-message">(Render Error)</span>';
        } finally {
            markdownUpdateTimeoutId = null; // Clear ID after execution
        }
    };

    if (force) {
        performUpdate(); // Execute immediately if forced
    } else {
        markdownUpdateTimeoutId = setTimeout(performUpdate, MARKDOWN_UPDATE_INTERVAL_MS); // Schedule throttled update
    }
}

function appendSources(messageElement, sources) {
    if (!messageElement || !sources || sources.length === 0 || messageElement.querySelector('.sources')) return; // Don't add if empty or already added
    const sourcesDiv = document.createElement('div');
    sourcesDiv.classList.add('sources');
    sourcesDiv.innerHTML = '<strong>Sources:</strong>';
    const sourcesList = document.createElement('ul');
    sources.forEach(source => {
        if (source && source.source) {
            const item = document.createElement('li');
            item.classList.add('source-item');
            const link = document.createElement('span'); // Use span, click handled by chatbox listener
            link.classList.add('source-link');
            link.textContent = source.source;
            link.dataset.filename = source.source; // Store filename in data attribute
            link.title = `Click to view ${source.source}`;
            item.appendChild(link);
            sourcesList.appendChild(item);
        }
    });
    sourcesDiv.appendChild(sourcesList);
    messageElement.appendChild(sourcesDiv);
}

function showLoading(show) {
    if(sendButton) sendButton.disabled = show;
    if(userInput) userInput.disabled = show;
    if (!show && userInput) userInput.focus();
}

function displayError(message, targetElement = null) {
    const errorContent = `<span class="error-message">Error: ${message}</span>`;
    if (targetElement) {
        const contentEl = targetElement.querySelector('.message-content') || targetElement;
        if(contentEl) contentEl.innerHTML = errorContent;
    } else {
        addMessage(`Error: ${message}`, 'assistant');
    }
    scrollToBottom();
    isStreaming = false; // Ensure streaming stops on error display
    if (markdownUpdateTimeoutId) clearTimeout(markdownUpdateTimeoutId);
}

// --- Send Message and Handle Response ---
async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;

    addMessage(query, 'user');
    userInput.value = '';
    showLoading(true);

    // Reset state for the new message
    currentAssistantMessageElement = null;
    accumulatedResponseForMarkdown = '';
    sourcesForCurrentMessage = [];
    isStreaming = false;
    if (markdownUpdateTimeoutId) { clearTimeout(markdownUpdateTimeoutId); markdownUpdateTimeoutId = null; }

    try {
        const response = await fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: query }), });

        if (!response.ok) { const errorData = await response.json().catch(() => ({ error: `Request failed (${response.status})` })); throw new Error(errorData.error || `Server error (${response.status})`); }

        const contentType = response.headers.get("content-type");

        // Handle JSON Response (non-streaming actions)
        if (contentType && contentType.includes("application/json")) {
            const result = await response.json();
            console.log("Received JSON:", result);
            if (result.status === 'success' && result.message) {
                addMessage(result.message, 'assistant');
                if(result.themes) { // Handle theme updates specifically
                    window.themesData = result.themes; // Update global theme data (assuming script.js defines it)
                    if(typeof populateThemeCircles === 'function') populateThemeCircles();
                    if(result.new_theme_key && typeof applyTheme === 'function') { applyTheme(result.new_theme_key); }
                }
            } else if (result.status === 'error' && result.message) { displayError(result.message); }
            else { addMessage("Received an unexpected response.", 'assistant'); }
            showLoading(false);
            return;
        }
        // Handle SSE Stream
        else if (contentType && contentType.includes("text/event-stream")) {
            if (!response.body) throw new Error("Stream body missing.");
            currentAssistantMessageElement = addMessage('', 'assistant');
            if (!currentAssistantMessageElement) throw new Error("Failed message element.");
            isStreaming = true;
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let firstTokenReceived = false;

            while (true) {
                const { value, done } = await reader.read();
                if (done) { isStreaming = false; scheduleMarkdownUpdate(true); console.log("Stream finished."); break; }
                buffer += decoder.decode(value, { stream: true });
                let boundaryIndex;
                while ((boundaryIndex = buffer.indexOf('\n\n')) >= 0) {
                    const message = buffer.substring(0, boundaryIndex);
                    buffer = buffer.substring(boundaryIndex + 2);
                    let eventType = 'message', eventData = '';
                    message.split('\n').forEach(line => { if (line.startsWith('event:')) eventType = line.substring(6).trim(); else if (line.startsWith('data:')) eventData = line.substring(5).trim(); });
                    if (!firstTokenReceived && (eventType === 'token' || eventType === 'sources')) { const contentEl = currentAssistantMessageElement?.querySelector('.message-content'); if (contentEl) contentEl.innerHTML = ''; firstTokenReceived = true; }
                    if (eventType === 'sources') { try { sourcesForCurrentMessage = JSON.parse(eventData); console.log("Sources:", sourcesForCurrentMessage); } catch (e) { console.error("Bad sources:", e, eventData); } }
                    else if (eventType === 'token') { try { const data = JSON.parse(eventData); if (data.token) { accumulatedResponseForMarkdown += data.token; scheduleMarkdownUpdate(); } } catch (e) { console.error("Bad token:", e, eventData); } }
                    else if (eventType === 'end') { console.log("End event."); }
                    else if (eventType === 'error') { try { const data = JSON.parse(eventData); throw new Error(data.error || "Stream error"); } catch (e) { console.error("Bad error event:", e, eventData); throw new Error("Unparsable stream error."); } }
                }
            }
        } else { throw new Error(`Unexpected response content type: ${contentType}`); }
    } catch (error) {
        console.error('Error during sendMessage:', error);
        displayError(error.message, currentAssistantMessageElement || null);
        isStreaming = false;
        if (markdownUpdateTimeoutId) clearTimeout(markdownUpdateTimeoutId);
    } finally {
        if (!isStreaming) { showLoading(false); }
    }
}

// --- File Viewer Functions ---
async function viewFile(filename) {
    if (!filename || !fileViewer) return;
    console.log(`Requesting file: ${filename}`);
    if(fileViewerTitle) fileViewerTitle.textContent = `Loading ${filename}...`;
    if(fileViewerContent) { fileViewerContent.innerHTML = ''; fileViewerContent.textContent = ''; fileViewerContent.className = ''; }
    fileViewer.style.width = ''; // Reset width
    fileViewer.classList.add('visible');

    try {
        const encodedFilename = encodeURIComponent(filename);
        const response = await fetch(`/get_guide_content?filename=${encodedFilename}`);
        if (!response.ok) { const errorData = await response.json().catch(() => ({ description: `HTTP error ${response.status}` })); throw new Error(errorData.description || `Could not load file (${response.status})`); }
        const data = await response.json();
        if(fileViewerTitle) fileViewerTitle.textContent = data.filename;
        if(fileViewerContent) {
            fileViewerContent.className = ''; // Reset classes again
            if (data.is_html && data.html_content) { fileViewerContent.innerHTML = data.html_content; fileViewerContent.classList.add('viewer-is-html'); }
            else if (!data.is_html && data.text_content) { fileViewerContent.textContent = data.text_content; fileViewerContent.classList.add('viewer-is-text'); }
            else { throw new Error("Received empty or invalid content."); }

            // Auto-Resize Logic
            requestAnimationFrame(() => {
                try {
                    const contentScrollWidth = fileViewerContent.scrollWidth; const currentViewerWidth = fileViewer.offsetWidth; const styles = getComputedStyle(fileViewerContent); const internalPadding = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight); const scrollbarWidth = fileViewerContent.offsetWidth - fileViewerContent.clientWidth - (parseFloat(styles.borderLeftWidth) || 0) - (parseFloat(styles.borderRightWidth) || 0); const buffer = 40; let targetWidth = contentScrollWidth + internalPadding + scrollbarWidth + buffer; const sideMenu = document.getElementById('side-menu'); const sideMenuWidth = sideMenu ? sideMenu.offsetWidth : 0; const minMainWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--min-main-view-width')) || 400; const gap = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--resize-gap')) || 15; const maxWidth = window.innerWidth - sideMenuWidth - minMainWidth - gap; const minWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--viewer-min-width')) || 300; targetWidth = Math.max(minWidth, Math.min(targetWidth, maxWidth)); const defaultMinWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--viewer-default-width').match(/clamp\(([^,]+)/)?.[1] || '300px');
                    if (targetWidth > currentViewerWidth || targetWidth < defaultMinWidth) { fileViewer.style.transition = 'none'; fileViewer.style.width = `${targetWidth}px`; fileViewer.offsetHeight; fileViewer.style.transition = ''; }
                } catch (measureError) { console.error("Error during viewer auto-resize measurement:", measureError); }
            });
            fileViewerContent.scrollTop = 0; // Scroll to top
        }
    } catch (error) {
        console.error('View file error:', error);
        if(fileViewerTitle) fileViewerTitle.textContent = `Error`;
        if(fileViewerContent) { fileViewerContent.textContent = `Could not load file: ${filename}\n\nError: ${error.message}`; fileViewerContent.className = 'viewer-is-text'; }
    }
}

function closeViewer() {
    if(fileViewer) fileViewer.classList.remove('visible');
    if(fileViewer) fileViewer.style.width = ''; // Reset width
}

// --- Viewer Resizing Functions ---
function startResizeViewer(e) {
    if (!fileViewer) return;
    isResizingViewer = true;
    viewerStartX = e.clientX;
    viewerStartWidth = fileViewer.offsetWidth;
    fileViewer.classList.add('resizing');
    document.addEventListener('mousemove', doResizeViewer);
    document.addEventListener('mouseup', stopResizeViewer);
    document.body.style.userSelect = 'none';
    e.preventDefault();
}

function doResizeViewer(e) {
    if (!isResizingViewer || !fileViewer) return;
    const dx = e.clientX - viewerStartX;
    let newWidth = viewerStartWidth - dx;
    const sideMenu = document.getElementById('side-menu');
    const sideMenuWidth = sideMenu ? sideMenu.offsetWidth : 0;
    const minMainWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--min-main-view-width')) || 400;
    const gap = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--resize-gap')) || 15;
    const maxWidth = window.innerWidth - sideMenuWidth - minMainWidth - gap;
    const minWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--viewer-min-width')) || 300;
    newWidth = Math.max(minWidth, Math.min(newWidth, maxWidth));
    fileViewer.style.width = `${newWidth}px`;
}

function stopResizeViewer() {
    if (isResizingViewer) {
        isResizingViewer = false;
        if (fileViewer) fileViewer.classList.remove('resizing');
        document.removeEventListener('mousemove', doResizeViewer);
        document.removeEventListener('mouseup', stopResizeViewer);
        document.body.style.userSelect = '';
    }
}

// --- Initialization Function for Chat Module ---
function initializeChat() {
    console.log("Initializing Chat Module...");
    // Add Chat specific listeners
    if(sendButton) sendButton.addEventListener('click', sendMessage);
    if(userInput) userInput.addEventListener('keypress', function(event) { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); } });
    if(chatbox) chatbox.addEventListener('click', (event) => { let target = event.target; while (target && target !== chatbox) { if (target.classList.contains('source-link')) { const filename = target.dataset.filename; if (filename) { viewFile(filename); return; } } target = target.parentNode; } });
    if(closeViewerBtn) closeViewerBtn.addEventListener('click', closeViewer);
    if (viewerResizer) { viewerResizer.addEventListener('mousedown', startResizeViewer); }
    else { console.warn("Viewer resizer element not found."); }
}