// static/js/script.js
// Base script: Handles shared elements, theme, view switching, and initializes modules.

// --- Embedded Theme Data ---
let themesData = typeof embeddedThemesData !== 'undefined' ? embeddedThemesData : {};
const defaultThemeName = typeof embeddedDefaultThemeName !== 'undefined' ? embeddedDefaultThemeName : 'default';

// --- Embedded Profile Data ---
let profilesData = typeof embeddedProfilesData !== 'undefined' ? embeddedProfilesData : {};
const defaultProfileKey = typeof embeddedDefaultProfileKey !== 'undefined' ? embeddedDefaultProfileKey : 'default';
let selectedProfileKey = defaultProfileKey; // State variable for currently selected profile
let allLoadedGuides = typeof embeddedAllGuides !== 'undefined' ? embeddedAllGuides : []; // Store all loaded guide names

// --- Shared DOM Elements ---
const themeCirclesContainer = document.getElementById('theme-circles-container');
const chatInterfaceView = document.getElementById('chat-interface-view');
// REMOVED crmInterfaceView
const showChatBtn = document.getElementById('show-chat-btn');
// REMOVED showCrmBtn
const menuNavButtons = document.querySelectorAll('#side-menu .menu-nav-button'); // Keep for potential future use / styling active
const profileSelect = document.getElementById('profile-select'); // New profile select element
const profileGuidesListContainer = document.getElementById('profile-guides-list'); // New guides list UL

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

// --- Profile Functions ---
function populateProfileDropdown() {
    if (!profilesData || Object.keys(profilesData).length === 0 || !profileSelect) {
        console.error("No profile data or select element available.");
        return;
    }

    profileSelect.innerHTML = ''; // Clear existing options

    // Add options from profilesData
    for (const key in profilesData) {
        if (profilesData.hasOwnProperty(key)) {
            const profile = profilesData[key];
            const option = document.createElement('option');
            option.value = key;
            option.textContent = profile.name || key;
            profileSelect.appendChild(option);
        }
    }

    // Set the selected profile based on localStorage or default
    const savedProfileKey = localStorage.getItem('selectedProfileKey');
    if (savedProfileKey && profilesData.hasOwnProperty(savedProfileKey)) {
        selectedProfileKey = savedProfileKey;
    } else {
        selectedProfileKey = defaultProfileKey;
    }

    profileSelect.value = selectedProfileKey;
    console.log(`Initial profile set to: ${selectedProfileKey}`);

    // Add event listener for profile changes
    profileSelect.addEventListener('change', handleProfileChange);

    // Update the guides list for the initial profile
    updateProfileGuidesList();
}

function handleProfileChange() {
    if (profileSelect) {
        selectedProfileKey = profileSelect.value;
        console.log(`Profile changed to: ${selectedProfileKey}`);
        try {
            localStorage.setItem('selectedProfileKey', selectedProfileKey);
        } catch (e) {
            console.warn("Could not save profile preference:", e);
        }
        // Update the displayed guides list
        updateProfileGuidesList();
        // Optionally, add logic here to clear chat or show a message about the profile change
        // For now, it just updates the state and storage.
    }
}

// CORRECTED: Function to update the displayed list of guides for the selected profile
function updateProfileGuidesList() {
    if (!profileGuidesListContainer || !profilesData) {
        console.error("Profile guides list container or profiles data not available.");
        return;
    }

    profileGuidesListContainer.innerHTML = ''; // Clear current list

    const profile = profilesData[selectedProfileKey];
    if (!profile) {
        profileGuidesListContainer.innerHTML = '<li class="muted">Error: Profile not found</li>';
        return;
    }

    const profileGuides = profile.available_guides;

    // Determine which list of guides to display
    // If profileGuides is empty, use the globally loaded allLoadedGuides list
    const guidesToDisplay = (!profileGuides || profileGuides.length === 0) ? allLoadedGuides : profileGuides;

    if (!guidesToDisplay || guidesToDisplay.length === 0) {
        // This message now appears if *no* guides were loaded at all, or if a profile
        // specifically lists an empty array AND no guides were loaded globally.
        profileGuidesListContainer.innerHTML = '<li class="muted">(No guides available/loaded)</li>';
    } else {
        guidesToDisplay.forEach(guideFilename => {
            const li = document.createElement('li');
            const link = document.createElement('span'); // Use span for styling and click handling
            link.classList.add('profile-guide-link');
            link.textContent = guideFilename;
            link.dataset.filename = guideFilename; // Store filename for click handler
            link.title = `Click to view ${guideFilename}`;
            li.appendChild(link);
            profileGuidesListContainer.appendChild(li);
        });
    }
}


// Function to get the currently selected profile key (used by chat.js)
function getSelectedProfileKey() {
    return selectedProfileKey;
}


// --- REMOVED View Switching Function ---
// No longer needed as there's only one main view.

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize Themes
    populateThemeCircles();
    const savedTheme = localStorage.getItem('selectedTheme');
    if (savedTheme && themesData[savedTheme]) { applyTheme(savedTheme); }
    else if (defaultThemeName && themesData[defaultThemeName]) { applyTheme(defaultThemeName); }
    else if (Object.keys(themesData).length > 0) { applyTheme(Object.keys(themesData)[0]); }

    // 2. Initialize Profiles and Dropdown (which now also updates the guide list)
    populateProfileDropdown();

    // 3. Set Initial View (Chat view is now the only view)
    if(chatInterfaceView) chatInterfaceView.classList.remove('view-hidden');
    if(showChatBtn) showChatBtn.classList.add('active'); // Keep chat button active

    // 4. REMOVED Main Navigation Listeners for switching views

    // 5. Initialize Chat Module (Add listeners defined in chat.js)
    if (typeof initializeChat === 'function') {
        initializeChat();
    } else {
        console.error("initializeChat function not found in chat.js");
    }

    // 6. Add event listener for clicks on profile guides list (Event Delegation)
    if (profileGuidesListContainer) {
        profileGuidesListContainer.addEventListener('click', (event) => {
            const target = event.target;
            // Check if the clicked element is a guide link
            if (target && target.classList.contains('profile-guide-link') && target.dataset.filename) {
                const filename = target.dataset.filename;
                console.log(`Clicked profile guide link: ${filename}`);
                // Check if viewFile function exists (it should be global from chat.js)
                if (typeof viewFile === 'function') {
                    viewFile(filename);
                } else {
                    console.error("viewFile function not found. Ensure chat.js is loaded.");
                }
            }
        });
    }


    // 7. REMOVED CRM Module Initialization

    console.log("Main application initialized.");
});