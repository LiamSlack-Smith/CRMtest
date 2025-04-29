// static/js/script.js
// Base script: Handles shared elements, theme, view switching, and initializes modules.

// --- Embedded Theme Data ---
let themesData = typeof embeddedThemesData !== 'undefined' ? embeddedThemesData : {};
const defaultThemeName = typeof embeddedDefaultThemeName !== 'undefined' ? embeddedDefaultThemeName : 'default';

// --- Shared DOM Elements ---
const themeCirclesContainer = document.getElementById('theme-circles-container');
const chatInterfaceView = document.getElementById('chat-interface-view');
const crmInterfaceView = document.getElementById('crm-interface-view');
const showChatBtn = document.getElementById('show-chat-btn');
const showCrmBtn = document.getElementById('show-crm-btn');
const menuNavButtons = document.querySelectorAll('.menu-nav-button'); // All nav buttons

// --- Theme Functions ---
function applyTheme(themeKey) {
    const theme = themesData[themeKey];
    if (!theme) { console.error(`Theme "${themeKey}" not found.`); return; }
    console.log(`Applying theme: ${theme.name || themeKey}`);
    const root = document.documentElement;
    for (const [key, value] of Object.entries(theme)) {
        if (key !== 'name') { root.style.setProperty(`--${key}`, value); }
    }
    try { localStorage.setItem('selectedTheme', themeKey); }
    catch (e) { console.warn("Could not save theme preference:", e); }
    if (themeCirclesContainer) {
        const circles = themeCirclesContainer.querySelectorAll('.theme-circle');
        circles.forEach(circle => {
            circle.classList.toggle('selected', circle.dataset.themeKey === themeKey);
        });
    }
 }
function populateThemeCircles() {
    if (!themesData || Object.keys(themesData).length === 0 || !themeCirclesContainer) {
        console.error("No theme data or container available.");
        return;
    }
    themeCirclesContainer.innerHTML = '';
    for (const [key, theme] of Object.entries(themesData)) {
        const circle = document.createElement('div');
        circle.classList.add('theme-circle');
        circle.dataset.themeKey = key;
        circle.title = theme.name || key;
        const colorLeft = theme['secondary-bg'] || '#444';
        const colorRight = theme['accent-color'] || '#0d6efd';
        const gradientLeft = `linear-gradient(to right, ${colorLeft} 0%, ${colorLeft} 50%, transparent 50.1%)`;
        const gradientRight = `linear-gradient(to right, transparent 49.9%, ${colorRight} 50%, ${colorRight} 100%)`;
        circle.style.backgroundImage = `${gradientLeft}, ${gradientRight}`;
        circle.addEventListener('click', () => { applyTheme(key); });
        themeCirclesContainer.appendChild(circle);
    }
 }

// --- View Switching Function ---
function switchToAppView(viewToShowId) {
    const views = [chatInterfaceView, crmInterfaceView];
    views.forEach(view => {
        if (view) view.classList.toggle('view-hidden', view.id !== viewToShowId);
    });
    // Update main navigation button active state
    menuNavButtons.forEach(button => {
         if (button.id.startsWith('show-')) {
            button.classList.toggle('active', button.id === `show-${viewToShowId.split('-')[0]}-btn`);
         }
    });
    console.log(`Switched to app view: ${viewToShowId}`);

    // If switching to CRM view, reset it to dashboard (handled by CRM button listener)
    // If switching away from CRM, potentially clear CRM state if needed (not currently necessary)
}


// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Themes
    populateThemeCircles();
    const savedTheme = localStorage.getItem('selectedTheme');
    if (savedTheme && themesData[savedTheme]) { applyTheme(savedTheme); }
    else if (defaultThemeName && themesData[defaultThemeName]) { applyTheme(defaultThemeName); }
    else if (Object.keys(themesData).length > 0) { applyTheme(Object.keys(themesData)[0]); }

    // 2. Set Initial View
    switchToAppView('chat-interface-view');
    // Initialize CRM view state (call function from crm.js)
    if (typeof initializeCrmView === 'function') {
        initializeCrmView(); // Sets CRM to dashboard initially
    } else {
        console.error("initializeCrmView function not found in crm.js");
    }


    // 3. Add Main Navigation Listeners
    if(showChatBtn) showChatBtn.addEventListener('click', () => switchToAppView('chat-interface-view'));
    if(showCrmBtn) showCrmBtn.addEventListener('click', () => {
        switchToAppView('crm-interface-view');
        // Reset CRM view to dashboard when switching to the CRM tab
        if (typeof showCrmSection === 'function') {
            showCrmSection('crm-dashboard-view');
            // Deactivate all CRM nav buttons
            const crmNavs = document.querySelectorAll('#side-menu .menu-section:nth-child(3) .menu-nav-button');
            crmNavs.forEach(btn => btn.classList.remove('active'));
            // Update CRM title
             const crmViewTitle = document.getElementById('crm-view-title');
             if(crmViewTitle) crmViewTitle.textContent = `CRM Dashboard`;

        } else {
             console.error("showCrmSection function not found in crm.js");
        }
    });

    // 4. Initialize Chat Module (Add listeners defined in chat.js)
    if (typeof initializeChat === 'function') {
        initializeChat();
    } else {
        console.error("initializeChat function not found in chat.js");
    }

    // 5. Initialize CRM Module (Add listeners defined in crm.js)
     if (typeof initializeCrm === 'function') {
        initializeCrm();
    } else {
        console.error("initializeCrm function not found in crm.js");
    }

    console.log("Main application initialized.");
});