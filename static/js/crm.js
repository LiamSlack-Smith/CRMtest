// static/js/crm.js
// Handles CRM view switching, data fetching, rendering, and actions.

// --- CRM DOM Elements (Declare with let, assign in initializeCrm) ---
let crmViewTitle, crmListView, crmListTitle, crmRecordListContainer, crmDetailView,
    crmDetailTitle, crmDetailContent, crmDashboardView, crmNavContactsBtn,
    crmNavSupportItemsBtn, crmNavCasesBtn, crmNavClaimItemsBtn, crmBackBtn,
    crmNewRecordBtn, crmEditBtn, crmSaveBtn, crmCancelBtn, crmDeleteBtn,
    crmDetailForm, createRelatedButtonContainer, relatedRecordsContainer,
    relatedRecordsTitle, relatedRecordsList;

// --- CRM State Variables ---
let currentCrmView = 'dashboard';
let currentCrmRecordType = null;
let currentCrmRecordId = null;
let currentRecordData = null; // Store fetched data for edit/cancel

// --- Constants ---
// Defined here or fetched/passed if dynamic
const VALID_CONTACT_TYPES = ["client", "bequestor", "donor", "relative", "representative", "volunteer", "veteran", "family"];
const VALID_SUPPORT_CATEGORIES = ["housing", "wellbeing", "claims", "employment", "other"];
const VALID_LEGISLATIONS = ["DRCA", "VEA", "MRCA"];


// Configure marked.js - Ensure marked is loaded before this script
if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
} else {
    console.error("marked.js library not loaded before crm.js");
}

// --- CRM View Switching ---
function showCrmSection(sectionId) {
     // Ensure elements are assigned before using them
     const sections = [crmDashboardView, crmListView, crmDetailView];
     sections.forEach(section => { if (section) section.classList.toggle('view-hidden', section.id !== sectionId); });
     currentCrmView = sectionId.replace('crm-', '').replace('-view', '');
     if(crmNewRecordBtn) crmNewRecordBtn.classList.toggle('view-hidden', currentCrmView !== 'list');
     console.log(`Switched CRM section to: ${currentCrmView}`);
}

// --- CRM Data Display Functions ---
async function displayRecordList(recordType) {
    console.log(`Fetching list for: ${recordType}`);
    currentCrmRecordType = recordType; currentCrmRecordId = null; currentRecordData = null;
    // Check if elements are ready
    if (!crmRecordListContainer || !crmListTitle || !crmViewTitle) { console.error("CRM List view elements not found or not initialized"); return; }
    crmRecordListContainer.innerHTML = '<p class="muted loading">Loading...</p>';
    const recordTypeName = recordType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    crmListTitle.textContent = `All ${recordTypeName}s`;
    crmViewTitle.textContent = `${recordTypeName} List`;
    showCrmSection('crm-list-view');
    // Highlight active nav button
    [crmNavContactsBtn, crmNavSupportItemsBtn, crmNavCasesBtn, crmNavClaimItemsBtn].forEach(btn => {
        if(btn) btn.classList.toggle('active', btn.id === `nav-${recordType.replace('_', '-')}-btn`);
    });

    try {
        const response = await fetch(`/api/${recordType}`);
        if (!response.ok) throw new Error(`Failed fetch ${recordType} (${response.status})`);
        const records = await response.json();
        crmRecordListContainer.innerHTML = '';
        if (records.length === 0) { crmRecordListContainer.innerHTML = `<p class="muted">No ${recordTypeName.toLowerCase()}s found.</p>`; return; }
        const ul = document.createElement('ul'); ul.classList.add('record-list');
        records.forEach(record => {
            const li = document.createElement('li'); li.classList.add('record-list-item'); li.dataset.id = record.id; li.dataset.type = recordType;
            let displayInfo = ''; let displayName = '';
            if (recordType === 'contacts') { displayName = record.name || `${record.first_name || ''} ${record.last_name || ''}`.trim() || `Contact #${record.id}`; displayInfo = `<span>${record.email || 'No Email'}</span>`; }
            else if (recordType === 'support_items') { displayName = `${record.category?.toUpperCase() || 'Item'} (${record.sub_type || 'N/A'}) #${record.id}`; displayInfo = `<span>Status: ${record.status || 'N/A'} | Contact: ${record.contact_id}</span>`; }
            else if (recordType === 'cases') { displayName = record.case_name || record.summary || `Case #${record.id}`; if (displayName.length > 40) displayName = displayName.substring(0, 37) + '...'; displayInfo = `<span>Type: ${record.type || 'N/A'} | Item: ${record.support_item_id}</span>`; }
            else if (recordType === 'claim_items') { displayName = record.condition || `Claim Item #${record.id}`; if (displayName.length > 40) displayName = displayName.substring(0, 37) + '...'; displayInfo = `<span>Class: ${record.classification || 'N/A'} | Case: ${record.case_id}</span>`; }
            else { displayName = `Record #${record.id}`; displayInfo = `<span>Type: ${recordType}</span>`; }
            li.innerHTML = `<strong>${displayName}</strong> ${displayInfo}`; ul.appendChild(li);
        });
        crmRecordListContainer.appendChild(ul);
    } catch (error) { console.error(`Error fetching ${recordType}:`, error); crmRecordListContainer.innerHTML = `<p class="error-message">Error loading ${recordType}.</p>`; }
}

// --- UPDATED renderDetailView ---
function renderDetailView(record, mode = 'view') {
     // Check all required elements, now including the title/list for related records
     if (!crmDetailView || !crmDetailTitle || !crmDetailContent || !createRelatedButtonContainer || !relatedRecordsContainer || !relatedRecordsTitle || !relatedRecordsList) {
         console.error("CRM Detail view elements (including related placeholders/title/list) not found or not initialized. Aborting render.");
         // Display error to user?
         if(crmDetailContent) crmDetailContent.innerHTML = '<p class="error-message">Error: UI components missing. Cannot display details.</p>';
         return; // Exit if essential elements are missing
     }

     const recordTypeNameSingle = currentCrmRecordType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
     let viewTitleName = record.name || record.case_name || record.condition || record.summary || record.id || '';
     crmDetailTitle.textContent = mode === 'edit' ? (record.id ? `Edit ${recordTypeNameSingle}` : `New ${recordTypeNameSingle}`) : `View ${recordTypeNameSingle}: ${viewTitleName}`;
     crmDetailView.dataset.mode = mode;
     crmDetailContent.querySelector('.form-error')?.remove();
     crmDetailContent.innerHTML = '';

     let detailHtml = '<div class="detail-grid">';
     let fields = [];

     // --- Field Definitions ---
     if (currentCrmRecordType === 'contacts') { fields = [ { key: 'first_name', label: 'First Name', type: 'text', required: true }, { key: 'last_name', label: 'Last Name', type: 'text', required: true }, { key: 'name', label: 'Full Name (Display)', type: 'readonly', viewOnly: true}, { key: 'email', label: 'Email', type: 'email' }, { key: 'phone', label: 'Phone', type: 'tel' }, { key: 'address', label: 'Address', type: 'textarea' }, { key: 'gender', label: 'Gender', type: 'text' }, { key: 'date_of_birth', label: 'Date of Birth', type: 'date' }, { key: 'is_atsi', label: 'Aboriginal/TSI', type: 'checkbox' }, { key: 'contact_types', label: 'Contact Types', type: 'checkbox-group', options: VALID_CONTACT_TYPES }, { key: 'service_status', label: 'Service Status', type: 'text' }, { key: 'service_branch', label: 'Service Branch', type: 'text' }, { key: 'service_enlistment_date', label: 'Enlistment Date', type: 'date' }, { key: 'service_discharge_date', label: 'Discharge Date', type: 'date' }, { key: 'service_role', label: 'Role', type: 'text' }, { key: 'service_deployment_summary', label: 'Deployment Summary', type: 'textarea' }, { key: 'service_discharge_reason', label: 'Discharge Reason', type: 'text' }, { key: 'personal_marital_status', label: 'Marital Status', type: 'text' }, { key: 'personal_spouse_name', label: 'Spouse/Partner Name', type: 'text' }, { key: 'personal_has_children', label: 'Has Children', type: 'checkbox' }, { key: 'personal_children_living_with', label: 'Children Living With', type: 'checkbox' }, { key: 'personal_cald', label: 'CALD', type: 'checkbox' }, { key: 'pmkeys_number', label: 'PMKeys Number', type: 'text' }, { key: 'service_card_pension_details', label: 'Service Card/Pension', type: 'text' }, { key: 'dva_number', label: 'DVA Number', type: 'text' }, ]; if (mode === 'view') { fields.push( { key: 'created_at', label: 'Created', type: 'readonly' }, { key: 'updated_at', label: 'Updated', type: 'readonly' }, { key: 'support_item_count', label: 'Support Items', type: 'readonly' }, { key: 'case_count', label: 'Cases', type: 'readonly' }, { key: 'claim_item_count', label: 'Claim Items', type: 'readonly' } ); } }
     else if (currentCrmRecordType === 'support_items') { fields = [ { key: 'contact_id', label: 'Contact ID', type: 'number', required: true, readonly: true }, { key: 'category', label: 'Category', type: 'select', required: true, options: VALID_SUPPORT_CATEGORIES }, { key: 'sub_type', label: 'Sub Type', type: 'text' }, { key: 'status', label: 'Status', type: 'text', required: true }, { key: 'description', label: 'Description', type: 'textarea' }, { key: 'housing_assistance_type', label: 'Housing Assistance Type', type: 'text' }, ]; if (mode === 'view') { fields.push( { key: 'created_at', label: 'Created', type: 'readonly' }, { key: 'updated_at', label: 'Updated', type: 'readonly' }, { key: 'case_count', label: 'Cases', type: 'readonly' } ); } }
     else if (currentCrmRecordType === 'cases') { fields = [ { key: 'support_item_id', label: 'Support Item ID', type: 'number', required: true, readonly: true }, { key: 'contact_id', label: 'Contact ID', type: 'number', readonly: true, viewOnly: true }, { key: 'case_name', label: 'Case Name', type: 'text' }, { key: 'summary', label: 'Summary', type: 'text', required: true }, { key: 'case_area', label: 'Case Area', type: 'text' }, { key: 'type', label: 'Type', type: 'text', required: true }, { key: 'sub_type', label: 'Sub Type', type: 'text', required: true }, { key: 'legislation', label: 'Legislation', type: 'select', options: VALID_LEGISLATIONS, allowEmpty: true }, { key: 'claim_type', label: 'Claim Type', type: 'text' }, { key: 'waiting_on_client', label: 'Waiting On Client', type: 'checkbox' }, { key: 'consent_received_date', label: 'Consent Received', type: 'date' }, { key: 'self_submitted', label: 'Self Submitted', type: 'checkbox' }, { key: 'paperwork_due_date', label: 'Paperwork Due', type: 'date' }, { key: 'paperwork_received', label: 'Paperwork Received', type: 'checkbox' }, { key: 'paperwork_recurring_reminder', label: 'Recurring Reminder', type: 'checkbox' }, { key: 'description', label: 'Description', type: 'textarea' }, { key: 'details', label: 'Internal Details', type: 'textarea' }, { key: 'h4h_approval_team', label: 'H4H Approval Team', type: 'text' }, { key: 'h4h_send_initial_approval', label: 'H4H Send Initial Approval', type: 'checkbox' }, { key: 'h4h_initial_approved', label: 'H4H Initial Approved', type: 'checkbox' }, { key: 'h4h_approved_by', label: 'H4H Approved By', type: 'text' }, { key: 'allocated_property', label: 'Allocated Property', type: 'text' }, { key: 'assigned_case_manager', label: 'Assigned Case Manager', type: 'text' }, ]; if (mode === 'view') { fields.push( { key: 'created_at', label: 'Created', type: 'readonly' }, { key: 'updated_at', label: 'Updated', type: 'readonly' }, { key: 'claim_item_count', label: 'Claim Items', type: 'readonly'} ); } }
     else if (currentCrmRecordType === 'claim_items') { fields = [ { key: 'case_id', label: 'Case ID', type: 'number', required: true, readonly: true }, { key: 'contact_id', label: 'Contact ID', type: 'number', required: true, readonly: true }, { key: 'condition', label: 'Condition', type: 'text', required: true }, { key: 'classification', label: 'Classification', type: 'text' }, { key: 'sop', label: 'SOP', type: 'text' }, { key: 'reasonable_hypothesis', label: 'Reasonable Hypothesis', type: 'text' }, { key: 'legislations', label: 'Legislation(s)', type: 'checkbox-group', options: VALID_LEGISLATIONS }, { key: 'medical_date_identification', label: 'Medical Date (ID)', type: 'date' }, { key: 'medical_date_diagnosis', label: 'Medical Date (Diagnosis)', type: 'date' }, { key: 'determined_by_admin', label: 'Determined By Admin', type: 'text' }, { key: 'rejection_reason', label: 'Rejection Reason', type: 'text' }, { key: 'determined_by', label: 'Determined By', type: 'text' }, ]; if (mode === 'view') { fields.push( { key: 'created_at', label: 'Created', type: 'readonly' }, { key: 'updated_at', label: 'Updated', type: 'readonly' } ); } }
     else { fields = []; detailHtml += `<p>Details/Edit for ${currentCrmRecordType} not implemented yet.</p>`; }

     // --- Generate HTML for fields ---
     fields.forEach(field => { if (mode === 'edit' && field.viewOnly) return; const value = record[field.key] ?? (field.type === 'checkbox-group' ? [] : ''); let displayValue = value; if (field.type === 'date' && value) { try { displayValue = new Date(value + 'T00:00:00Z').toLocaleDateString(); } catch(e) { displayValue = value; } } else if (field.type === 'checkbox') { displayValue = value ? 'Yes' : 'No'; } else if (Array.isArray(value)) { displayValue = value.join(', '); } const requiredAttr = field.required ? 'required' : ''; const readonlyAttr = field.readonly ? 'readonly' : ''; const escapedValue = (typeof value === 'string') ? value.replace(/"/g, '"') : value; const inputId = `detail-${field.key}`; detailHtml += `<div class="detail-field" data-field-name="${field.key}">`; detailHtml += `<label for="${inputId}">${field.label}${field.required ? ' *' : ''}</label>`; detailHtml += `<span class="view-mode">${displayValue !== '' && displayValue !== null && (!Array.isArray(displayValue) || displayValue.length > 0) ? displayValue : 'N/A'}</span>`; if (mode === 'edit' && !field.viewOnly) { if (field.readonly) { detailHtml += `<input type="text" id="${inputId}" name="${field.key}" class="edit-mode readonly-field" value="${escapedValue}" readonly disabled>`; } else if (field.type === 'textarea') { detailHtml += `<textarea id="${inputId}" name="${field.key}" class="edit-mode" rows="3" ${requiredAttr}>${escapedValue}</textarea>`; } else if (field.type === 'select' && field.options) { detailHtml += `<select id="${inputId}" name="${field.key}" class="edit-mode" ${requiredAttr}>`; if (!field.required || field.allowEmpty) detailHtml += `<option value="">-- Select --</option>`; field.options.forEach(opt => { const selected = value === opt ? ' selected' : ''; const displayOpt = opt.charAt(0).toUpperCase() + opt.slice(1); detailHtml += `<option value="${opt}"${selected}>${displayOpt}</option>`; }); detailHtml += `</select>`; } else if (field.type === 'checkbox') { const checkedAttr = value ? 'checked' : ''; detailHtml += `<input type="checkbox" id="${inputId}" name="${field.key}" class="edit-mode" value="true" ${checkedAttr} ${requiredAttr}>`; } else if (field.type === 'checkbox-group' && field.options) { detailHtml += `<div class="edit-mode checkbox-group">`; field.options.forEach(opt => { const checked = Array.isArray(value) && value.includes(opt) ? 'checked' : ''; const cbId = `${inputId}-${opt}`; detailHtml += `<span class="checkbox-option"><input type="checkbox" id="${cbId}" name="${field.key}" value="${opt}" ${checked}> <label for="${cbId}">${opt}</label></span>`; }); detailHtml += `</div>`; } else { detailHtml += `<input type="${field.type || 'text'}" id="${inputId}" name="${field.key}" class="edit-mode" value="${field.type === 'date' ? value : escapedValue}" ${requiredAttr}>`; } } else if (mode === 'edit' && field.readonly) { detailHtml += `<span class="edit-mode readonly-field">${displayValue !== '' && displayValue !== null && (!Array.isArray(displayValue) || displayValue.length > 0) ? displayValue : 'N/A'}</span>`; } detailHtml += `</div>`; });
     detailHtml += '</div>';
     crmDetailContent.innerHTML = detailHtml;

     // --- Render "Create Related" Button ---
     createRelatedButtonContainer.innerHTML = ''; createRelatedButtonContainer.classList.add('view-hidden');
     if (mode === 'view' && record.id) {
         let relatedType = null; let buttonText = ''; let parentContactId = null;
         if (currentCrmRecordType === 'contacts') { relatedType = 'support_items'; buttonText = 'New Support Item'; }
         else if (currentCrmRecordType === 'support_items') { relatedType = 'cases'; buttonText = 'New Case'; }
         else if (currentCrmRecordType === 'cases') { relatedType = 'claim_items'; buttonText = 'New Claim Item'; parentContactId = record.contact_id; }
         if (relatedType) { const button = document.createElement('button'); button.type = 'button'; button.id = 'create-related-btn'; button.classList.add('btn-primary'); button.textContent = buttonText; button.dataset.relatedType = relatedType; button.dataset.parentId = record.id; button.dataset.parentType = currentCrmRecordType; if (parentContactId) { button.dataset.contactId = parentContactId; } createRelatedButtonContainer.appendChild(button); createRelatedButtonContainer.classList.remove('view-hidden'); }
     }

     // --- Render Related Records Section ---
     relatedRecordsContainer.classList.add('view-hidden'); relatedRecordsList.innerHTML = '';
     if (mode === 'view' && record.id) {
         let relatedEntityType = null; let relatedTitle = '';
         if (currentCrmRecordType === 'contacts') { relatedEntityType = 'support_items'; relatedTitle = 'Related Support Items'; }
         else if (currentCrmRecordType === 'support_items') { relatedEntityType = 'cases'; relatedTitle = 'Related Cases'; }
         else if (currentCrmRecordType === 'cases') { relatedEntityType = 'claim_items'; relatedTitle = 'Related Claim Items'; }
         if (relatedEntityType) { relatedRecordsTitle.textContent = relatedTitle; relatedRecordsContainer.classList.remove('view-hidden'); displayRelatedRecords(relatedEntityType, record.id); }
     }

     // --- Toggle Action Buttons ---
     crmEditBtn?.classList.toggle('view-hidden', mode === 'edit' || !record.id);
     crmDeleteBtn?.classList.toggle('view-hidden', mode === 'edit' || !record.id);
     crmSaveBtn?.classList.toggle('view-hidden', mode === 'view');
     crmCancelBtn?.classList.toggle('view-hidden', mode === 'view');
     crmBackBtn?.classList.toggle('view-hidden', mode === 'edit');
}

// --- Display Related Records ---
async function displayRelatedRecords(relatedType, parentId) {
    if (!relatedRecordsList) return;
    relatedRecordsList.innerHTML = '<p class="muted loading">Loading related records...</p>';
    let filterParam = '';
    if (currentCrmRecordType === 'contacts') filterParam = `contact_id=${parentId}`;
    else if (currentCrmRecordType === 'support_items') filterParam = `support_item_id=${parentId}`;
    else if (currentCrmRecordType === 'cases') filterParam = `case_id=${parentId}`;
    else { relatedRecordsList.innerHTML = '<p class="error-message">Unknown parent type.</p>'; return; }

    try {
        const response = await fetch(`/api/${relatedType}?${filterParam}`);
        if (!response.ok) throw new Error(`Failed fetch ${relatedType} (${response.status})`);
        const records = await response.json();
        relatedRecordsList.innerHTML = '';
        if (records.length === 0) { relatedRecordsList.innerHTML = `<p class="muted">No related ${relatedType.replace('_', ' ')} found.</p>`; return; }
        const ul = document.createElement('ul'); ul.classList.add('record-list');
        records.forEach(record => {
            const li = document.createElement('li'); li.classList.add('record-list-item', 'related-list-item'); li.dataset.id = record.id; li.dataset.type = relatedType;
            let displayInfo = ''; let displayName = '';
            if (relatedType === 'support_items') { displayName = `${record.category?.toUpperCase() || 'Item'} #${record.id}`; displayInfo = `<span>Status: ${record.status || 'N/A'}</span>`; }
            else if (relatedType === 'cases') { displayName = record.case_name || record.summary || `Case #${record.id}`; if (displayName.length > 30) displayName = displayName.substring(0, 27) + '...'; displayInfo = `<span>Type: ${record.type || 'N/A'}</span>`; }
            else if (relatedType === 'claim_items') { displayName = record.condition || `Claim Item #${record.id}`; if (displayName.length > 30) displayName = displayName.substring(0, 27) + '...'; displayInfo = `<span>Case: ${record.case_id}</span>`; }
            else { displayName = `Record #${record.id}`; }
            li.innerHTML = `<strong>${displayName}</strong> ${displayInfo}`; ul.appendChild(li);
        });
        relatedRecordsList.appendChild(ul);
    } catch (error) { console.error(`Error fetching related ${relatedType}:`, error); relatedRecordsList.innerHTML = `<p class="error-message">Error loading related ${relatedType}.</p>`; }
}


async function displayRecordDetail(recordType, recordId) {
    console.log(`Fetching detail for ${recordType} ID: ${recordId}`);
    currentCrmRecordId = recordId; currentCrmRecordType = recordType;
    if (!crmDetailContent || !crmDetailTitle || !crmViewTitle) return;
    crmDetailContent.innerHTML = '<p class="muted loading">Loading details...</p>';
    if(createRelatedButtonContainer) createRelatedButtonContainer.innerHTML = ''; createRelatedButtonContainer.classList.add('view-hidden');
    if(relatedRecordsContainer) relatedRecordsContainer.classList.add('view-hidden'); if(relatedRecordsList) relatedRecordsList.innerHTML = '';

    const recordTypeNameSingle = recordType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    crmDetailTitle.textContent = `View ${recordTypeNameSingle}`; crmViewTitle.textContent = `Record Detail`;
    showCrmSection('crm-detail-view');
    try {
         const response = await fetch(`/api/${recordType}/${recordId}`);
         if (!response.ok) { if(response.status === 404) throw new Error(`${recordTypeNameSingle} not found.`); throw new Error(`Failed fetch details (${response.status})`); }
         const record = await response.json();
         currentRecordData = record;
         renderDetailView(record, 'view');
    } catch (error) {
         console.error(`Error fetching detail:`, error);
         crmDetailContent.innerHTML = `<p class="error-message">${error.message}</p>`;
         crmDetailTitle.textContent = `Error`; currentRecordData = null;
         crmEditBtn?.classList.add('view-hidden'); crmDeleteBtn?.classList.add('view-hidden');
    }
}

// --- CRM Action Handlers ---
// --- UPDATED handleNewRecord ---
function handleNewRecord(event, parentContext = null) {
     let recordTypeToCreate = currentCrmRecordType;
     let prefillData = {};

     if (parentContext) {
         recordTypeToCreate = parentContext.relatedType;
         const parentIdField = parentContext.parentType === 'contacts' ? 'contact_id'
                             : parentContext.parentType === 'support_items' ? 'support_item_id'
                             : parentContext.parentType === 'cases' ? 'case_id'
                             : null;
         if (parentIdField) {
             prefillData[parentIdField] = parentContext.parentId;
             // Pre-fill contact_id for claim items using contactId from button dataset
             if (parentContext.parentType === 'cases' && recordTypeToCreate === 'claim_items' && parentContext.contactId) {
                 prefillData['contact_id'] = parentContext.contactId;
             }
         }
     }

     if (!recordTypeToCreate) { alert("Select record type from the menu first or use a 'New Related' button."); return; }

     console.log(`Showing New ${recordTypeToCreate} form`, prefillData);
     currentCrmRecordType = recordTypeToCreate;
     currentCrmRecordId = null;
     currentRecordData = { ...prefillData }; // Start with pre-filled data

     renderDetailView(currentRecordData, 'edit');
     showCrmSection('crm-detail-view');
     const recordTypeNameSingle = recordTypeToCreate.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
     crmViewTitle.textContent = `New ${recordTypeNameSingle}`;
     crmDetailTitle.textContent = `New ${recordTypeNameSingle}`;
}


async function handleSaveRecord(event) {
     event.preventDefault();
     if (!currentCrmRecordType || !crmDetailForm) return;
     const form = crmDetailForm;
     const dataToSave = {}; let isValid = true;
     const fieldElements = form.querySelectorAll('.detail-field');

     fieldElements.forEach(fieldDiv => {
         const fieldName = fieldDiv.dataset.fieldName;
         // Find the input/select/textarea/div.checkbox-group within this specific fieldDiv
         const inputElement = fieldDiv.querySelector('.edit-mode:not(.readonly-field)'); // Exclude readonly spans/inputs
         const labelElement = fieldDiv.querySelector('label');
         const isRequired = labelElement && labelElement.textContent.includes('*');

         if (inputElement) {
            let value;
            // Handle checkbox group (list fields)
            if (inputElement.classList.contains('checkbox-group')) {
                const checkedBoxes = inputElement.querySelectorAll(`input[type="checkbox"][name="${fieldName}"]:checked`);
                value = Array.from(checkedBoxes).map(cb => cb.value);
                if (isRequired && value.length === 0) { isValid = false; inputElement.style.border = '1px solid var(--error-color)'; }
                else { inputElement.style.border = ''; }
                dataToSave[fieldName] = value;
            }
            // Handle single checkbox
            else if (inputElement.type === 'checkbox') {
                value = inputElement.checked;
                if (isRequired && !value) { /* Optional validation */ }
                else { /* Reset style */ }
                dataToSave[fieldName] = value;
            }
            // Handle other input types
            else {
                value = inputElement.value.trim();
                 if (isRequired && !value) { isValid = false; inputElement.style.borderColor = 'var(--error-color)'; }
                 else { inputElement.style.borderColor = ''; }
                 if (inputElement.type === 'number' && value !== '') { dataToSave[fieldName] = parseInt(value, 10); }
                 // Include readonly fields (like pre-filled IDs) only if they have value
                 else if (inputElement.hasAttribute('readonly') || inputElement.hasAttribute('disabled')) {
                     if(inputElement.name && inputElement.value) {
                         // Attempt conversion back to number if needed
                         const originalType = currentRecordData?.[fieldName] !== undefined ? typeof currentRecordData[fieldName] : 'string';
                         if (originalType === 'number') {
                             dataToSave[fieldName] = parseInt(inputElement.value, 10);
                         } else {
                             dataToSave[fieldName] = inputElement.value;
                         }
                     }
                 }
                 else { dataToSave[fieldName] = value; }
            }
         } else {
             // If no editable input, check if there's a readonly field (like pre-filled ID)
             const readonlyInputElement = fieldDiv.querySelector('.edit-mode.readonly-field');
             if (readonlyInputElement && readonlyInputElement.name && readonlyInputElement.value) {
                 // Convert back to number if needed
                 const originalType = currentRecordData?.[fieldName] !== undefined ? typeof currentRecordData[fieldName] : 'string';
                 if (originalType === 'number') {
                     dataToSave[fieldName] = parseInt(readonlyInputElement.value, 10);
                 } else {
                     dataToSave[fieldName] = readonlyInputElement.value;
                 }
             }
         }
     });


     if (!isValid) { alert("Please fill in all required fields (*)."); return; }

     const isNew = !currentCrmRecordId;
     const url = isNew ? `/api/${currentCrmRecordType}` : `/api/${currentCrmRecordType}/${currentCrmRecordId}`;
     const method = isNew ? 'POST' : 'PUT';

     console.log(`Saving ${currentCrmRecordType}: ${method} ${url}`, dataToSave);
     if(crmSaveBtn) { crmSaveBtn.textContent = 'Saving...'; crmSaveBtn.disabled = true; }
     if(crmCancelBtn) crmCancelBtn.disabled = true;
     crmDetailContent.querySelector('.form-error')?.remove();

     try {
         const response = await fetch(url, { method: method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(dataToSave) });
         const result = await response.json();
         if (!response.ok) { throw new Error(result.error || `Save failed (${response.status})`); }
         console.log("Save successful:", result);
         // If editing, go back to the detail view of the saved record
         if (!isNew && result.id) {
             displayRecordDetail(currentCrmRecordType, result.id);
         } else if (isNew && result.id) { // If new, go back to list view of the type created
             displayRecordList(currentCrmRecordType);
         } else { // Fallback
              displayRecordList(currentCrmRecordType || 'contacts'); // Default to contacts list if type unclear
         }
     } catch (error) {
         console.error("Save error:", error);
         const errorDiv = document.createElement('div'); errorDiv.className = 'error-message form-error';
         errorDiv.textContent = `Save failed: ${error.message}`;
         crmDetailContent.prepend(errorDiv);
     } finally {
          if(crmSaveBtn) { crmSaveBtn.textContent = 'Save'; crmSaveBtn.disabled = false; }
          if(crmCancelBtn) crmCancelBtn.disabled = false;
     }
}

function handleCancelEdit() {
     if (currentCrmRecordId && currentRecordData) { renderDetailView(currentRecordData, 'view'); }
     else if (currentCrmRecordType) { displayRecordList(currentCrmRecordType); }
     else { showCrmSection('crm-dashboard-view'); crmViewTitle.textContent = `CRM Dashboard`; }
}

async function handleDeleteRecord() {
    if (!currentCrmRecordType || !currentCrmRecordId) { alert("No record selected."); return; }
    const recordTypeNameSingle = currentCrmRecordType.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    if (!confirm(`Are you sure you want to delete this ${recordTypeNameSingle} (ID: ${currentCrmRecordId})? This may also delete related records.`)) return;
    const url = `/api/${currentCrmRecordType}/${currentCrmRecordId}`;
    console.log(`Deleting ${currentCrmRecordType} ID: ${currentCrmRecordId}`);
    crmDeleteBtn.disabled = true; crmEditBtn.disabled = true; crmBackBtn.disabled = true;
    crmDetailContent.querySelector('.form-error')?.remove();
    try {
        const response = await fetch(url, { method: 'DELETE' });
        const result = await response.json();
        if (!response.ok) { throw new Error(result.error || `Delete failed (${response.status})`); }
        console.log("Delete successful:", result.message);
        displayRecordList(currentCrmRecordType);
    } catch (error) {
        console.error("Delete error:", error);
        const errorDiv = document.createElement('div'); errorDiv.className = 'error-message form-error';
        errorDiv.textContent = `Delete failed: ${error.message}`;
        crmDetailContent.prepend(errorDiv);
        crmDeleteBtn.disabled = false; crmEditBtn.disabled = false; crmBackBtn.disabled = false;
    }
}


// --- Initialization Function for CRM Module (UPDATED) ---
function initializeCrmView() {
    showCrmSection('crm-dashboard-view');
    if (crmViewTitle) crmViewTitle.textContent = `CRM Dashboard`;
}

function initializeCrm() {
    console.log("Initializing CRM Module...");

    // Assign DOM elements now that DOM is loaded
    crmViewTitle = document.getElementById('crm-view-title');
    crmListView = document.getElementById('crm-list-view');
    crmListTitle = document.getElementById('crm-list-title');
    crmRecordListContainer = document.getElementById('crm-record-list-container');
    crmDetailView = document.getElementById('crm-detail-view');
    crmDetailTitle = document.getElementById('crm-detail-title');
    crmDetailContent = document.getElementById('crm-detail-content');
    crmDashboardView = document.getElementById('crm-dashboard-view');
    crmNavContactsBtn = document.getElementById('nav-contacts-btn');
    crmNavSupportItemsBtn = document.getElementById('nav-support-items-btn');
    crmNavCasesBtn = document.getElementById('nav-cases-btn');
    crmNavClaimItemsBtn = document.getElementById('nav-claim-items-btn');
    crmBackBtn = document.getElementById('crm-back-btn');
    crmNewRecordBtn = document.getElementById('crm-new-record-btn');
    crmEditBtn = document.getElementById('crm-edit-btn');
    crmSaveBtn = document.getElementById('crm-save-btn');
    crmCancelBtn = document.getElementById('crm-cancel-btn');
    crmDeleteBtn = document.getElementById('crm-delete-btn');
    crmDetailForm = document.getElementById('crm-detail-form');
    createRelatedButtonContainer = document.getElementById('create-related-button-container');
    relatedRecordsContainer = document.getElementById('related-records-container');
    relatedRecordsTitle = document.getElementById('related-records-title');
    relatedRecordsList = document.getElementById('related-records-list');

    // --- DEBUG: Log assigned elements ---
    console.log("CRM Elements Assigned:", { crmViewTitle, crmListView, crmListTitle, crmRecordListContainer, crmDetailView, crmDetailTitle, crmDetailContent, crmDashboardView, crmNavContactsBtn, crmNavSupportItemsBtn, crmNavCasesBtn, crmNavClaimItemsBtn, crmBackBtn, crmNewRecordBtn, crmEditBtn, crmSaveBtn, crmCancelBtn, crmDeleteBtn, crmDetailForm, createRelatedButtonContainer, relatedRecordsContainer, relatedRecordsTitle, relatedRecordsList });
    // --- END DEBUG ---


    // CRM Data Nav Listeners
    if(crmNavContactsBtn) crmNavContactsBtn.addEventListener('click', () => displayRecordList('contacts'));
    if(crmNavSupportItemsBtn) crmNavSupportItemsBtn.addEventListener('click', () => displayRecordList('support_items'));
    if(crmNavCasesBtn) crmNavCasesBtn.addEventListener('click', () => displayRecordList('cases'));
    if(crmNavClaimItemsBtn) crmNavClaimItemsBtn.addEventListener('click', () => displayRecordList('claim_items'));

    // CRM List Item Click Listener (Event Delegation on main list container)
    if(crmRecordListContainer) crmRecordListContainer.addEventListener('click', (event) => {
        const listItem = event.target.closest('.record-list-item');
        if (listItem?.dataset.id && listItem?.dataset.type) {
            displayRecordDetail(listItem.dataset.type, parseInt(listItem.dataset.id, 10));
        }
    });

     // CRM Detail View Event Delegation (for related items and create button)
     if (crmDetailView) {
        crmDetailView.addEventListener('click', (event) => {
            // Handle clicks on related list items
            const relatedListItem = event.target.closest('.related-list-item');
            if (relatedListItem?.dataset.id && relatedListItem?.dataset.type) {
                console.log(`Clicked related ${relatedListItem.dataset.type} ID: ${relatedListItem.dataset.id}`);
                displayRecordDetail(relatedListItem.dataset.type, parseInt(relatedListItem.dataset.id, 10));
                return;
            }

            // Handle click on "Create Related" button
            const createButton = event.target.closest('#create-related-btn');
            if (createButton?.dataset.relatedType && createButton?.dataset.parentId) {
                 console.log(`Clicked create related ${createButton.dataset.relatedType} for parent ${createButton.dataset.parentType} ID: ${createButton.dataset.parentId}`);
                 const context = {
                     relatedType: createButton.dataset.relatedType,
                     parentId: parseInt(createButton.dataset.parentId, 10),
                     parentType: createButton.dataset.parentType,
                     contactId: createButton.dataset.contactId ? parseInt(createButton.dataset.contactId, 10) : null // Pass contactId if available
                 };
                 handleNewRecord(null, context);
                 return;
            }
        });
     }


    // CRM Detail Action Button Listeners
    if(crmBackBtn) crmBackBtn.addEventListener('click', () => {
        if (currentCrmRecordType) { displayRecordList(currentCrmRecordType); }
        else { showCrmSection('crm-dashboard-view'); crmViewTitle.textContent = `CRM Dashboard`; }
    });
    if(crmNewRecordBtn) crmNewRecordBtn.addEventListener('click', () => handleNewRecord(null)); // No context needed here
    if(crmEditBtn) crmEditBtn.addEventListener('click', () => { if(currentRecordData) renderDetailView(currentRecordData, 'edit'); });
    if(crmCancelBtn) crmCancelBtn.addEventListener('click', handleCancelEdit);
    if(crmDeleteBtn) crmDeleteBtn.addEventListener('click', handleDeleteRecord);

    // CRM Form Submission
    if(crmDetailForm) crmDetailForm.addEventListener('submit', handleSaveRecord);
}