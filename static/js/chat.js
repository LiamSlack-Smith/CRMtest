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
let currentAssistantContentElement = null; // Store the content div directly
let accumulatedResponseForMarkdown = ''; // Accumulate for dynamic rendering
let sourcesForCurrentMessage = [];
let markdownUpdateTimeoutId = null; // Timer ID for throttled markdown updates
let isStreaming = false;
// Adjust the interval as needed for responsiveness vs. performance
const MARKDOWN_UPDATE_INTERVAL_MS = 50; // Interval for updating markdown during streaming (throttling)
let isResizingViewer = false;
let viewerStartX = 0;
let viewerStartWidth = 0;

// --- Chat Helper Functions ---
function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    messageDiv.classList.add(sender === 'user' ? 'user-message' : 'assistant-message');

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');

    if (sender === 'user') {
        // Sanitize user input before inserting as HTML
        const tempDiv = document.createElement('div');
        tempDiv.textContent = text;
        contentDiv.innerHTML = tempDiv.innerHTML.replace(/\n/g, '<br>');
    } else {
        // Assistant message starts empty or with cursor
        contentDiv.innerHTML = text || '<span class="blinking-cursor">▍</span>'; // Initial state
        // Store references for streaming
        currentAssistantMessageElement = messageDiv;
        currentAssistantContentElement = contentDiv;
        accumulatedResponseForMarkdown = ''; // Reset accumulator
    }
    messageDiv.appendChild(contentDiv);

    if (chatbox) {
        chatbox.appendChild(messageDiv);
        scrollToBottom();
    } else {
        console.error("Chatbox element not found!");
    }
    return messageDiv; // Return the main message element
}

function scrollToBottom() {
    if (!chatbox) return;
    // Scroll down if user is already near the bottom
    const isScrolledToBottom = chatbox.scrollHeight - chatbox.clientHeight <= chatbox.scrollTop + 60; // Threshold
    if (isScrolledToBottom) {
        if ('scrollBehavior' in document.documentElement.style) {
            // Use 'auto' during streaming for faster, less jarring scrolls
            chatbox.scrollTo({ top: chatbox.scrollHeight, behavior: isStreaming ? 'auto' : 'smooth' });
        } else {
            chatbox.scrollTop = chatbox.scrollHeight; // Fallback for older browsers
        }
    }
}

// Function to schedule or force Markdown update
function scheduleMarkdownUpdate(force = false) {
    // If not streaming or forcing, clear any pending update and perform immediately
    if (!isStreaming || force) {
        if (markdownUpdateTimeoutId) {
            clearTimeout(markdownUpdateTimeoutId);
            markdownUpdateTimeoutId = null;
        }
        performMarkdownUpdate(force); // Pass 'force' state to the update function
    } else if (!markdownUpdateTimeoutId) {
        // If streaming and no update is scheduled, schedule one
        markdownUpdateTimeoutId = setTimeout(() => {
            performMarkdownUpdate(false); // Perform update, not final
        }, MARKDOWN_UPDATE_INTERVAL_MS);
    }
}

// Function to perform the actual Markdown update
function performMarkdownUpdate(isFinal = false) {
    if (!currentAssistantContentElement) {
        markdownUpdateTimeoutId = null;
        return;
    }

    try {
        // Parse the accumulated text
        let htmlContent = marked.parse(accumulatedResponseForMarkdown);

        // Add blinking cursor if streaming and it's not the final update
        if (isStreaming && !isFinal) {
             // Add cursor after the parsed HTML
             htmlContent += '<span class="blinking-cursor">▍</span>';
        }

        // Update the element's content
        currentAssistantContentElement.innerHTML = htmlContent;

        // If it's the final update, append sources and clear state
        if (isFinal) {
            appendSources(currentAssistantMessageElement, sourcesForCurrentMessage);
            // Clear references and state
            currentAssistantMessageElement = null;
            currentAssistantContentElement = null;
            accumulatedResponseForMarkdown = '';
            isStreaming = false; // Ensure state is false
            showLoading(false); // Re-enable input
        }

        scrollToBottom(); // Scroll after updating content

    } catch (e) {
        console.error("Markdown update error:", e);
        // Display a render error message if parsing fails
        currentAssistantContentElement.innerHTML += '<br><span class="error-message">(Render Error)</span>';
        // If this was supposed to be final, still clear state
        if (isFinal) {
            currentAssistantMessageElement = null;
            currentAssistantContentElement = null;
            accumulatedResponseForMarkdown = '';
            isStreaming = false;
            showLoading(false);
        }
    } finally {
        // Clear the timeout ID after the update function runs
        // This is important whether it was scheduled or forced
        markdownUpdateTimeoutId = null;
    }
}


// Function to render final Markdown and append sources (kept for clarity, but scheduleMarkdownUpdate(true) does this)
function finalizeAssistantMessage() {
    // Call scheduleMarkdownUpdate with force=true
    scheduleMarkdownUpdate(true);
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
    messageElement.appendChild(sourcesDiv); // Append to the main message element
}

function showLoading(show) {
    if(sendButton) sendButton.disabled = show;
    if(userInput) userInput.disabled = show;
    if (!show && userInput) userInput.focus();
}

function displayError(message, targetElement = null) {
    const errorContent = `<span class="error-message">Error: ${message}</span>`;
    let messageToUpdate = targetElement || currentAssistantMessageElement; // Use current if target isn't specified

    if (messageToUpdate) {
        const contentEl = messageToUpdate.querySelector('.message-content');
        if (contentEl) {
             contentEl.innerHTML = errorContent; // Replace content with error
             // Remove sources if they were added before error
             const sourcesEl = messageToUpdate.querySelector('.sources');
             if(sourcesEl) sourcesEl.remove();
        } else {
             // Fallback if content element somehow missing
             messageToUpdate.innerHTML = errorContent;
        }
    } else {
        // If no assistant message exists yet, add a new one
        addMessage(`Error: ${message}`, 'assistant');
    }

    scrollToBottom();
    isStreaming = false; // Ensure streaming stops on error display
    // Clear potentially dangling references
    currentAssistantMessageElement = null;
    currentAssistantContentElement = null;
    accumulatedResponseForMarkdown = '';
    showLoading(false); // Ensure input is re-enabled
    if (markdownUpdateTimeoutId) { clearTimeout(markdownUpdateTimeoutId); markdownUpdateTimeoutId = null; } // Clear any pending markdown updates
}

// --- Send Message and Handle Response ---
async function sendMessage() {
    const query = userInput.value.trim();
    if (!query) return;

    addMessage(query, 'user');
    userInput.value = '';
    showLoading(true);

    // Reset state for the new message
    currentAssistantMessageElement = null; // Will be set by addMessage('','assistant')
    currentAssistantContentElement = null;
    accumulatedResponseForMarkdown = '';
    sourcesForCurrentMessage = [];
    isStreaming = false;
    if (markdownUpdateTimeoutId) { clearTimeout(markdownUpdateTimeoutId); markdownUpdateTimeoutId = null; } // Clear any previous timer

    // Get the selected profile key from script.js
    const profileKey = typeof getSelectedProfileKey === 'function' ? getSelectedProfileKey() : 'default';
    console.log("Sending message with profile:", profileKey);


    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // Include the profile_key in the request body
            body: JSON.stringify({ query: query, profile_key: profileKey }),
        });

        if (!response.ok) {
            // Try to parse error JSON, otherwise use status text
            let errorMsg = `Server error (${response.status})`;
            try {
                const errorData = await response.json();
                errorMsg = errorData.error || errorMsg;
            } catch (e) {
                 errorMsg = response.statusText || errorMsg;
            }
            throw new Error(errorMsg);
        }


        const contentType = response.headers.get("content-type");

        // Handle JSON Response (non-streaming actions like theme creation, support log)
        if (contentType && contentType.includes("application/json")) {
            const result = await response.json();
            console.log("Received JSON:", result);
            if (result.status === 'success' && result.message) {
                addMessage(result.message, 'assistant'); // Adds a *new* message bubble
            } else if (result.status === 'error' && result.message) {
                displayError(result.message); // Display error in a new bubble
            } else {
                addMessage("Received an unexpected response.", 'assistant');
            }
            showLoading(false); // Re-enable input after non-streaming response
            // Handle theme updates specifically after receiving JSON
            if(result.themes && typeof window !== 'undefined') {
                window.themesData = result.themes;
                if(typeof populateThemeCircles === 'function') populateThemeCircles();
                if(result.new_theme_key && typeof applyTheme === 'function') { applyTheme(result.new_theme_key); }
            }
            return;
        }
        // Handle SSE Stream (For RAG answers)
        else if (contentType && contentType.includes("text/event-stream")) {
            if (!response.body) throw new Error("Stream body missing.");

            // Create the initial assistant message bubble (empty or with cursor)
            addMessage('', 'assistant'); // This sets currentAssistantMessageElement and currentAssistantContentElement

            if (!currentAssistantContentElement) throw new Error("Failed to create assistant message element.");

            isStreaming = true;
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            let firstTokenReceived = false;

            while (true) {
                const { value, done } = await reader.read();
                if (done) {
                    isStreaming = false;
                    finalizeAssistantMessage(); // Render final markdown and sources
                    console.log("Stream finished.");
                    break; // Exit the loop
                }

                buffer += decoder.decode(value, { stream: true });
                let boundaryIndex;

                while ((boundaryIndex = buffer.indexOf('\n\n')) >= 0) {
                    const message = buffer.substring(0, boundaryIndex);
                    buffer = buffer.substring(boundaryIndex + 2);
                    let eventType = 'message', eventData = '';

                    message.split('\n').forEach(line => {
                        if (line.startsWith('event:')) eventType = line.substring(6).trim();
                        else if (line.startsWith('data:')) eventData = line.substring(5).trim();
                    });

                    // Process events
                    if (eventType === 'sources') {
                        try {
                            sourcesForCurrentMessage = JSON.parse(eventData);
                            console.log("Sources received:", sourcesForCurrentMessage);
                            // Don't append sources here, wait until the end
                        } catch (e) {
                            console.error("Bad sources JSON:", e, eventData);
                        }
                    } else if (eventType === 'token') {
                        try {
                            const data = JSON.parse(eventData);
                            if (data.token) {
                                if (!firstTokenReceived) {
                                    // Clear the initial blinking cursor on first token
                                    currentAssistantContentElement.innerHTML = '';
                                    firstTokenReceived = true;
                                }
                                // Append token directly to the content (as text node to prevent XSS)
                                currentAssistantContentElement.appendChild(document.createTextNode(data.token));
                                accumulatedResponseForMarkdown += data.token; // Also accumulate for final render
                                scheduleMarkdownUpdate(); // Schedule a throttled markdown update
                                scrollToBottom(); // Scroll as content is added
                            }
                        } catch (e) {
                            console.error("Bad token JSON:", e, eventData);
                        }
                    } else if (eventType === 'end') {
                        // The 'end' event might arrive before the reader signals 'done'
                        // We handle the finalization in the 'done' block above
                        console.log("End event received.");
                    } else if (eventType === 'error') {
                         // Handle error events within the stream
                         try {
                             const data = JSON.parse(eventData);
                             displayError(data.error || "Stream error reported by server", currentAssistantMessageElement);
                             reader.cancel(); // Cancel reading the rest of the stream
                             return; // Exit processing this response
                         } catch (e) {
                             console.error("Bad error event JSON:", e, eventData);
                             displayError("Unparsable stream error reported by server.", currentAssistantMessageElement);
                             reader.cancel(); // Cancel reading
                             return; // Exit
                         }
                    }
                } // end while boundaryIndex
            } // end while(true) reader loop
        } else {
            // Handle unexpected content type
             throw new Error(`Unexpected response content type: ${contentType}`);
        }
    } catch (error) {
        console.error('Error during sendMessage:', error);
        displayError(error.message, currentAssistantMessageElement || null); // Display error in current bubble if exists
        // Ensure state is reset even if error happens mid-stream
        isStreaming = false;
        currentAssistantMessageElement = null;
        currentAssistantContentElement = null;
        accumulatedResponseForMarkdown = '';
        showLoading(false);
        if (markdownUpdateTimeoutId) { clearTimeout(markdownUpdateTimeoutId); markdownUpdateTimeoutId = null; } // Clear any pending markdown updates
    }
    // Removed finally block as showLoading is handled by finalizeAssistantMessage or error handler
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
                    const contentScrollWidth = fileViewerContent.scrollWidth; const currentViewerWidth = fileViewer.offsetWidth; const styles = getComputedStyle(fileViewerContent); const internalPadding = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight); const scrollbarWidth = fileViewerContent.offsetWidth - fileViewerContent.clientWidth - (parseFloat(styles.borderLeftWidth) || 0) - (parseFloat(styles.borderRightWidth) || 0); const buffer = 40; let targetWidth = contentScrollWidth + internalPadding + scrollbarWidth + buffer; const sideMenu = document.getElementById('side-menu'); const sideMenuWidth = sideMenu ? sideMenu.offsetWidth : 0; const minMainWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--min-main-view-width')) || 400; const gap = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--resize-gap')) || 15; const maxWidth = window.innerWidth - sideMenuWidth - minMainWidth - gap; const minWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--viewer-min-width')) || 300; targetWidth = Math.max(minWidth, Math.min(targetWidth, maxWidth));
                    // CORRECTED: Access the captured group from the regex match
                    const defaultMinWidth = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--viewer-default-width').match(/clamp\(([^,]+)\)/)?.[1] || '300px');
                    // Only resize if needed, avoid transition flicker
                    if (Math.abs(targetWidth - currentViewerWidth) > 2) {
                       fileViewer.style.transition = 'none'; // Disable transition during resize
                       fileViewer.style.width = `${targetWidth}px`;
                       fileViewer.offsetHeight; // Force reflow
                       fileViewer.style.transition = ''; // Re-enable transition
                    }
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

    // Initialize marked - Ensure it's loaded
    if (typeof marked !== 'undefined') {
       marked.setOptions({
           breaks: true, // Convert single line breaks to <br>
           gfm: true,    // Enable GitHub Flavored Markdown
       });
       console.log("Marked.js initialized.");
    } else {
       console.error("Marked.js library not loaded.");
    }
}