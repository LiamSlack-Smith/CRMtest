# crm_logic.py
import os
import json
import threading
import datetime
import re
import logging
import enum # Import enum for SupportItemCategory

# --- Constants ---
CRM_DATA_FILE = "crm_data/crm_data.json" # Centralized path
DEFAULT_CRM_STRUCTURE = {
    "contacts": {},
    "support_items": {},
    "cases": {},
    "claim_items": {}, # Added Claim Items
    "next_contact_id": 1,
    "next_support_item_id": 1,
    "next_case_id": 1,
    "next_claim_item_id": 1 # Added Claim Item ID counter
}

# Define valid enum values
class SupportItemCategory(enum.Enum):
    HOUSING = "housing"
    WELLBEING = "wellbeing"
    CLAIMS = "claims"
    EMPLOYMENT = "employment"
    OTHER = "other"
VALID_SUPPORT_CATEGORIES = [e.value for e in SupportItemCategory]

# Define valid values based on documents
VALID_CONTACT_TYPES = ["client", "bequestor", "donor", "relative", "representative", "volunteer", "veteran", "family"]
VALID_LEGISLATIONS = ["DRCA", "VEA", "MRCA"] # Add others if known


# --- Lock for Thread Safety ---
crm_data_lock = threading.Lock()

# --- Custom Exceptions ---
class CrmBaseException(Exception):
    """Base class for CRM logic errors."""
    status_code = 500 # Default internal server error

class NotFoundError(CrmBaseException):
    """Resource not found."""
    status_code = 404

class ValidationError(CrmBaseException):
    """Invalid input data."""
    status_code = 400

class ConflictError(CrmBaseException):
    """Data conflict (e.g., duplicate email)."""
    status_code = 409

class StorageError(CrmBaseException):
    """Error during file read/write."""
    status_code = 500


# --- Data Handling Functions ---
def load_crm_data(filepath=CRM_DATA_FILE):
    """Loads CRM data from the specified JSON file (thread-safe)."""
    with crm_data_lock:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Validate structure - check for all expected keys from default
                if not isinstance(data, dict) or not all(k in data for k in DEFAULT_CRM_STRUCTURE.keys()):
                    logging.warning(f"Invalid or incomplete structure in {filepath}. Merging with default.")
                    # Merge loaded data with defaults to preserve existing data but add missing keys
                    merged_data = DEFAULT_CRM_STRUCTURE.copy()
                    if isinstance(data, dict): # Only merge if loaded data is a dict
                        for key in DEFAULT_CRM_STRUCTURE.keys():
                             if key in data:
                                 # Basic type check before assigning
                                 if isinstance(data[key], type(DEFAULT_CRM_STRUCTURE[key])):
                                     merged_data[key] = data[key]
                                 else:
                                     logging.warning(f"Type mismatch for key '{key}' in {filepath}. Using default.")
                    return merged_data
                return data
        except FileNotFoundError:
            logging.info(f"CRM data file '{filepath}' not found. Initializing.")
            return DEFAULT_CRM_STRUCTURE.copy() # Return a copy
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {filepath}. Returning default.", exc_info=True)
            return DEFAULT_CRM_STRUCTURE.copy()
        except Exception as e:
            logging.error(f"Error loading CRM data from {filepath}: {e}", exc_info=True)
            raise StorageError(f"Failed to load CRM data: {e}") from e

def save_crm_data(data, filepath=CRM_DATA_FILE):
    """Saves CRM data to the specified JSON file atomically (thread-safe)."""
    with crm_data_lock:
        temp_filepath = filepath + ".tmp"
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            os.replace(temp_filepath, filepath) # Atomic rename
            logging.debug(f"CRM data saved to {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error saving CRM data to {filepath}: {e}", exc_info=True)
            if os.path.exists(temp_filepath):
                try: os.remove(temp_filepath)
                except Exception as rm_err: logging.error(f"Error removing temp file {temp_filepath}: {rm_err}")
            raise StorageError(f"Failed to save CRM data: {e}") from e

def initialize_crm_storage():
    """Ensures the CRM data file exists and is valid on startup."""
    logging.info(f"Checking CRM JSON data file: {CRM_DATA_FILE}")
    try:
        initial_data = load_crm_data()
        save_crm_data(initial_data)
        logging.info("CRM JSON data file check/initialization complete.")
    except StorageError as e:
        logging.critical(f"Failed CRM storage initialization: {e}. CRM features may fail.", exc_info=True)


# --- Helper Functions ---
def _validate_bool(value):
    """Validates and converts input to boolean, allowing None for optional."""
    if value is None: return None # Allow clearing optional bools
    if isinstance(value, bool): return value
    if isinstance(value, str):
        if value.lower() in ['true', 'yes', '1']: return True
        if value.lower() in ['false', 'no', '0']: return False
    # Consider None or empty string as False for optional boolean fields? Or raise error?
    # Let's raise error for clarity if it's not explicitly boolean or known string values
    if value in [None, '']: return None # Allow clearing optional bools
    raise ValidationError(f"Invalid boolean value: '{value}'. Use true/false or yes/no.")

def _validate_date(value):
    """Validates YYYY-MM-DD date string, returns string or None."""
    if value is None or value == '': return None
    if not isinstance(value, str): raise ValidationError(f"Invalid date format: '{value}'. Expected string.")
    try:
        datetime.datetime.strptime(value, '%Y-%m-%d')
        return value # Return original string if valid date format
    except ValueError:
        raise ValidationError(f"Invalid date format: '{value}'. Use YYYY-MM-DD.")

def _validate_list_values(value_list, valid_options, field_name):
    """Validates that all items in a list are within the valid options."""
    if not isinstance(value_list, list): raise ValidationError(f"'{field_name}' must be a list.")
    invalid_items = [item for item in value_list if item not in valid_options]
    if invalid_items: raise ValidationError(f"Invalid values in '{field_name}': {', '.join(invalid_items)}. Valid options are: {', '.join(valid_options)}")
    return value_list # Return validated list

# --- CRM Logic Functions ---

# -- Contacts --
def create_contact_logic(data):
    """Logic to create a new contact."""
    # --- Field Extraction ---
    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()
    email = data.get('email')
    phone = data.get('phone')
    gender = data.get('gender')
    date_of_birth = data.get('date_of_birth')
    is_atsi = data.get('is_atsi', False) # Default false if omitted
    address = data.get('address')
    contact_types = data.get('contact_types', [])
    service_status = data.get('service_status')
    service_branch = data.get('service_branch')
    service_enlistment_date = data.get('service_enlistment_date')
    service_discharge_date = data.get('service_discharge_date')
    service_role = data.get('service_role')
    service_deployment_summary = data.get('service_deployment_summary')
    service_discharge_reason = data.get('service_discharge_reason')
    personal_marital_status = data.get('personal_marital_status')
    personal_spouse_name = data.get('personal_spouse_name')
    personal_has_children = data.get('personal_has_children', False)
    personal_children_living_with = data.get('personal_children_living_with', False)
    personal_cald = data.get('personal_cald', False)
    pmkeys_number = data.get('pmkeys_number')
    service_card_pension_details = data.get('service_card_pension_details')
    dva_number = data.get('dva_number')

    # --- Validation ---
    if not first_name and not last_name: raise ValidationError("First Name or Last Name is required.")
    name = f"{first_name} {last_name}".strip() # Combined name

    if email is not None:
        email = email.strip() if isinstance(email, str) else ''
        if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email): raise ValidationError("Invalid email format")
        email = email if email else None
    if phone is not None:
        phone = phone.strip() if isinstance(phone, str) else ''
        phone = phone if phone else None

    contact_types = _validate_list_values(contact_types, VALID_CONTACT_TYPES, 'contact_types')

    try: # Validate bools and dates
        is_atsi = _validate_bool(is_atsi)
        personal_has_children = _validate_bool(personal_has_children)
        personal_children_living_with = _validate_bool(personal_children_living_with)
        personal_cald = _validate_bool(personal_cald)
        date_of_birth = _validate_date(date_of_birth)
        service_enlistment_date = _validate_date(service_enlistment_date)
        service_discharge_date = _validate_date(service_discharge_date)
    except ValidationError as e: raise e # Re-raise validation errors

    # --- Uniqueness Check & Creation ---
    crm_data = load_crm_data()
    if email: # Check email uniqueness
        for contact in crm_data.get('contacts', {}).values():
            if contact.get('email') and contact['email'].lower() == email.lower():
                raise ConflictError(f"Contact with email '{email}' already exists.")

    new_id = crm_data['next_contact_id']
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

    new_contact_data = { # Store validated/cleaned values
        "id": new_id, "name": name, "email": email, "phone": phone,
        "first_name": first_name, "last_name": last_name,
        "gender": gender, "date_of_birth": date_of_birth, "is_atsi": is_atsi, "address": address,
        "contact_types": contact_types, "service_status": service_status, "service_branch": service_branch,
        "service_enlistment_date": service_enlistment_date, "service_discharge_date": service_discharge_date,
        "service_role": service_role, "service_deployment_summary": service_deployment_summary,
        "service_discharge_reason": service_discharge_reason, "personal_marital_status": personal_marital_status,
        "personal_spouse_name": personal_spouse_name, "personal_has_children": personal_has_children,
        "personal_children_living_with": personal_children_living_with, "personal_cald": personal_cald,
        "pmkeys_number": pmkeys_number, "service_card_pension_details": service_card_pension_details,
        "dva_number": dva_number,
        "created_at": now_iso, "updated_at": now_iso
    }
    crm_data['contacts'][str(new_id)] = new_contact_data
    crm_data['next_contact_id'] += 1

    save_crm_data(crm_data)
    logging.info(f"Created contact (JSON): ID={new_id}, Name='{name}'")
    return new_contact_data

def get_contacts_logic(search_query=None):
    """Logic to get a list of contacts, optionally filtered."""
    crm_data = load_crm_data()
    all_contacts = list(crm_data.get('contacts', {}).values())
    filtered_contacts = all_contacts

    if search_query:
        search_lower = search_query.lower()
        filtered_contacts = [
            c for c in all_contacts if (
                (c.get('name') and search_lower in c['name'].lower()) or
                (c.get('first_name') and search_lower in c['first_name'].lower()) or
                (c.get('last_name') and search_lower in c['last_name'].lower()) or
                (c.get('email') and search_lower in c['email'].lower()) or
                (c.get('phone') and search_lower in c['phone'].lower()) or
                (c.get('dva_number') and search_lower in c['dva_number'].lower())
            )
        ]

    sorted_contacts = sorted(filtered_contacts, key=lambda c: c.get('name', '').lower())
    # Return simplified list data
    return [{"id": c.get("id"), "name": c.get("name"), "email": c.get("email"), "phone": c.get("phone"), "created_at": c.get("created_at")} for c in sorted_contacts]

def get_contact_logic(contact_id):
    """Logic to get a single contact by ID."""
    crm_data = load_crm_data()
    contact = crm_data.get('contacts', {}).get(str(contact_id))
    if not contact: raise NotFoundError("Contact not found")

    # Calculate counts
    support_items = crm_data.get('support_items', {})
    cases = crm_data.get('cases', {})
    claim_items = crm_data.get('claim_items', {})
    contact_support_items = [item for item in support_items.values() if item.get('contact_id') == contact_id]
    contact_cases = [case for case in cases.values() if str(case.get('support_item_id')) in [str(item['id']) for item in contact_support_items]]
    contact_claim_items = [ci for ci in claim_items.values() if str(ci.get('case_id')) in [str(case['id']) for case in contact_cases]]

    contact_detail = contact.copy()
    contact_detail["support_item_count"] = len(contact_support_items)
    contact_detail["case_count"] = len(contact_cases)
    contact_detail["claim_item_count"] = len(contact_claim_items)
    return contact_detail

def update_contact_logic(contact_id, data):
    """Logic to update an existing contact."""
    contact_id_str = str(contact_id)
    crm_data = load_crm_data()
    contacts = crm_data.get('contacts', {})
    if contact_id_str not in contacts: raise NotFoundError("Contact not found")

    contact_to_update = contacts[contact_id_str]
    original_contact = contact_to_update.copy()

    # Define all updatable fields
    str_fields = [ 'phone', 'first_name', 'last_name', 'gender', 'address', 'service_status', 'service_branch', 'service_role', 'service_deployment_summary', 'service_discharge_reason', 'personal_marital_status', 'personal_spouse_name', 'pmkeys_number', 'service_card_pension_details', 'dva_number' ]
    date_fields = ['date_of_birth', 'service_enlistment_date', 'service_discharge_date']
    bool_fields = ['is_atsi', 'personal_has_children', 'personal_children_living_with', 'personal_cald']
    list_fields = ['contact_types']

    # Update string fields
    for field in str_fields:
        if field in data:
            value = data[field]
            if isinstance(value, str): value = value.strip()
            contact_to_update[field] = value if value is not None else '' # Store empty string if null

    # Recalculate combined name if first/last changed
    if 'first_name' in data or 'last_name' in data:
         first = contact_to_update.get('first_name','') # Use already updated value
         last = contact_to_update.get('last_name','')
         contact_to_update['name'] = f"{first} {last}".strip()

    # Update date fields
    for field in date_fields:
         if field in data:
             try: contact_to_update[field] = _validate_date(data[field])
             except ValidationError as e: raise ValidationError(f"Invalid date field '{field}': {e}")

    # Update boolean fields
    for field in bool_fields:
         if field in data:
             try: contact_to_update[field] = _validate_bool(data[field])
             except ValidationError as e: raise ValidationError(f"Invalid boolean field '{field}': {e}")

    # Update list fields
    for field in list_fields:
        if field in data:
            value = data[field]
            if field == 'contact_types': value = _validate_list_values(value, VALID_CONTACT_TYPES, field)
            else: # Add validation for other list fields if needed
                if not isinstance(value, list): raise ValidationError(f"'{field}' must be a list")
            contact_to_update[field] = value

    # Special handling for email uniqueness check
    if 'email' in data:
        email = data['email']
        if email is not None:
            email = email.strip() if isinstance(email, str) else ''; email = email if email else None
            if email and not re.match(r"[^@]+@[^@]+\.[^@]+", email): raise ValidationError("Invalid email format")
            # Check uniqueness excluding self
            for c_id, c_data in contacts.items():
                 if c_id != contact_id_str and c_data.get('email') and email and c_data['email'].lower() == email.lower():
                     raise ConflictError(f"The email '{email}' is already in use.")
            contact_to_update['email'] = email
        else: contact_to_update['email'] = None


    if contact_to_update != original_contact:
        contact_to_update['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_crm_data(crm_data)
        logging.info(f"Updated contact (JSON): ID={contact_id}")
        return contact_to_update
    else:
        return None # No changes

def delete_contact_logic(contact_id):
    """Logic to delete a contact and cascade."""
    contact_id_str = str(contact_id)
    crm_data = load_crm_data()
    if contact_id_str not in crm_data.get('contacts', {}): raise NotFoundError("Contact not found")

    # Cascade Deletion: Find all related items down the chain
    items_to_delete_ids = [item_id for item_id, item in crm_data.get('support_items', {}).items() if item.get('contact_id') == contact_id]
    cases_to_delete_ids = []
    claim_items_to_delete_ids = []

    for item_id_str_to_delete in items_to_delete_ids:
        item_id_int = int(item_id_str_to_delete) # Convert to int for case lookup
        # Find cases related to this support item
        related_case_ids = [case_id for case_id, case in crm_data.get('cases', {}).items() if case.get('support_item_id') == item_id_int] # Compare int IDs
        cases_to_delete_ids.extend(related_case_ids)
        # Find claim items related to these cases
        for case_id_str_to_delete in related_case_ids:
            case_id_int = int(case_id_str_to_delete)
            claim_items_to_delete_ids.extend([ci_id for ci_id, ci in crm_data.get('claim_items', {}).items() if ci.get('case_id') == case_id_int]) # Compare int IDs

    # Perform deletions (use sets to avoid duplicates)
    for ci_id in set(claim_items_to_delete_ids):
        if ci_id in crm_data.get('claim_items', {}): del crm_data['claim_items'][ci_id]; logging.debug(f"Deleting claim item {ci_id} (cascade)")
    for case_id in set(cases_to_delete_ids):
        if case_id in crm_data.get('cases', {}): del crm_data['cases'][case_id]; logging.debug(f"Deleting case {case_id} (cascade)")
    for item_id in items_to_delete_ids: # Already unique list
        if item_id in crm_data.get('support_items', {}): del crm_data['support_items'][item_id]; logging.debug(f"Deleting support item {item_id} (cascade)")

    # Delete the contact itself
    del crm_data['contacts'][contact_id_str]
    logging.debug(f"Deleting contact {contact_id}")

    save_crm_data(crm_data)
    logging.info(f"Deleted contact (JSON): ID={contact_id} and cascaded related items/cases.")
    return True

# -- Support Items --
def create_support_item_logic(data):
    """Logic to create a new support item."""
    contact_id = data.get('contact_id')
    category = data.get('category')
    description = data.get('description', '')
    status = data.get('status', 'Open')
    housing_assistance_type = data.get('housing_assistance_type')
    sub_type = data.get('sub_type')

    # Required field validation
    if not isinstance(contact_id, int) or contact_id <= 0: raise ValidationError("Missing or invalid 'contact_id'")
    if not category or category not in VALID_SUPPORT_CATEGORIES: raise ValidationError(f"Missing or invalid 'category'. Must be one of: {VALID_SUPPORT_CATEGORIES}")
    if not status or not isinstance(status, str): raise ValidationError("Missing or invalid 'status'")

    crm_data = load_crm_data()
    if str(contact_id) not in crm_data.get('contacts', {}): raise NotFoundError(f"Contact with ID {contact_id} not found.")

    new_id = crm_data['next_support_item_id']; now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    new_item_data = {
        "id": new_id, "contact_id": contact_id, "category": category, "status": status,
        "description": description, "housing_assistance_type": housing_assistance_type, "sub_type": sub_type,
        "created_at": now_iso, "updated_at": now_iso
    }
    crm_data['support_items'][str(new_id)] = new_item_data; crm_data['next_support_item_id'] += 1
    save_crm_data(crm_data)
    logging.info(f"Created support item (JSON): ID={new_id} for Contact ID={contact_id}")
    return new_item_data

def get_support_items_logic(filters):
    """Logic to get support items based on filters."""
    crm_data = load_crm_data()
    all_items = list(crm_data.get('support_items', {}).values())
    filtered_items = all_items

    # Apply filters only if they exist and are valid types
    if 'contact_id' in filters:
         contact_id = filters['contact_id']
         if isinstance(contact_id, int): filtered_items = [i for i in filtered_items if i.get('contact_id') == contact_id]
         else: raise ValidationError("Invalid 'contact_id' filter format.")
    if 'category' in filters:
        category = filters['category']
        if category in VALID_SUPPORT_CATEGORIES: filtered_items = [i for i in filtered_items if i.get('category') == category]
    if 'status' in filters:
        status = filters['status']
        if isinstance(status, str): filtered_items = [i for i in filtered_items if i.get('status', '').lower() == status.lower()]
    if 'sub_type' in filters:
        sub_type = filters['sub_type']
        if isinstance(sub_type, str): filtered_items = [i for i in filtered_items if i.get('sub_type', '').lower() == sub_type.lower()]

    return sorted(filtered_items, key=lambda item: item.get('created_at', ''), reverse=True)

def get_support_item_logic(item_id):
    """Logic to get a single support item."""
    crm_data = load_crm_data()
    item = crm_data.get('support_items', {}).get(str(item_id))
    if not item: raise NotFoundError("Support item not found")

    cases = crm_data.get('cases', {})
    count = sum(1 for case in cases.values() if case.get('support_item_id') == item_id)
    item_detail = item.copy()
    item_detail["case_count"] = count
    return item_detail

def update_support_item_logic(item_id, data):
    """Logic to update a support item."""
    item_id_str = str(item_id)
    crm_data = load_crm_data()
    items = crm_data.get('support_items', {})
    if item_id_str not in items: raise NotFoundError("Support item not found")

    item_to_update = items[item_id_str]
    original_item = item_to_update.copy()

    # Update fields if present in data
    if 'category' in data:
        category = data['category']
        if category not in VALID_SUPPORT_CATEGORIES: raise ValidationError(f"Invalid 'category'. Must be one of: {VALID_SUPPORT_CATEGORIES}")
        item_to_update['category'] = category
    if 'status' in data:
        status = data['status']
        if not isinstance(status, str): raise ValidationError("Invalid 'status'")
        item_to_update['status'] = status
    if 'description' in data:
        description = data['description']
        if not isinstance(description, str): raise ValidationError("Invalid 'description'")
        item_to_update['description'] = description
    if 'housing_assistance_type' in data: item_to_update['housing_assistance_type'] = data['housing_assistance_type']
    if 'sub_type' in data: item_to_update['sub_type'] = data['sub_type']

    if item_to_update != original_item:
        item_to_update['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_crm_data(crm_data)
        logging.info(f"Updated support item (JSON): ID={item_id}")
        return item_to_update
    else:
        return None # No changes

def delete_support_item_logic(item_id):
    """Logic to delete a support item and cascade cases and claim items."""
    item_id_str = str(item_id)
    crm_data = load_crm_data()
    if item_id_str not in crm_data.get('support_items', {}): raise NotFoundError("Support item not found")

    # Find related cases and their claim items
    cases_to_delete_ids = [case_id for case_id, case in crm_data.get('cases', {}).items() if case.get('support_item_id') == item_id]
    claim_items_to_delete_ids = []
    for case_id_str_to_delete in cases_to_delete_ids:
        case_id_int = int(case_id_str_to_delete)
        claim_items_to_delete_ids.extend([ci_id for ci_id, ci in crm_data.get('claim_items', {}).items() if ci.get('case_id') == case_id_int])

    # Perform deletions
    for ci_id in set(claim_items_to_delete_ids):
        if ci_id in crm_data.get('claim_items', {}): del crm_data['claim_items'][ci_id]; logging.debug(f"Deleting claim item {ci_id} (cascade)")
    for case_id in cases_to_delete_ids: # Already unique list
        if case_id in crm_data.get('cases', {}): del crm_data['cases'][case_id]; logging.debug(f"Deleting case {case_id} (cascade)")

    # Delete the support item itself
    del crm_data['support_items'][item_id_str]
    logging.debug(f"Deleting support item {item_id}")

    save_crm_data(crm_data)
    logging.info(f"Deleted support item (JSON): ID={item_id} and cascaded.")
    return True

# -- Cases --
def create_case_logic(data):
    """Logic to create a new case."""
    support_item_id = data.get('support_item_id')
    summary = data.get('summary')
    case_name = data.get('case_name', '')
    case_area = data.get('case_area')
    case_type = data.get('type')
    sub_type = data.get('sub_type')
    waiting_on_client = data.get('waiting_on_client', None)
    consent_received_date = data.get('consent_received_date')
    claim_type = data.get('claim_type')
    self_submitted = data.get('self_submitted', None)
    legislation = data.get('legislation')
    paperwork_due_date = data.get('paperwork_due_date')
    paperwork_received = data.get('paperwork_received', None)
    paperwork_recurring_reminder = data.get('paperwork_recurring_reminder', None)
    description = data.get('description', '')
    details = data.get('details', '')
    h4h_approval_team = data.get('h4h_approval_team')
    h4h_send_initial_approval = data.get('h4h_send_initial_approval', None)
    h4h_initial_approved = data.get('h4h_initial_approved', None)
    h4h_approved_by = data.get('h4h_approved_by')
    allocated_property = data.get('allocated_property')
    assigned_case_manager = data.get('assigned_case_manager')

    # --- Validation ---
    if not isinstance(support_item_id, int) or support_item_id <= 0: raise ValidationError("Missing or invalid 'support_item_id'")
    if not summary or not isinstance(summary, str) or not summary.strip(): raise ValidationError("Missing or invalid 'summary'")
    summary = summary.strip()
    if not case_type or not isinstance(case_type, str) or not case_type.strip(): raise ValidationError("Missing or invalid 'type' (case type)")
    case_type = case_type.strip()
    if not sub_type or not isinstance(sub_type, str) or not sub_type.strip(): raise ValidationError("Missing or invalid 'sub_type'")
    sub_type = sub_type.strip()
    # Validate legislation if provided and we have a list (Allow None/Empty)
    if legislation and isinstance(legislation, str) and legislation not in VALID_LEGISLATIONS:
         raise ValidationError(f"Invalid 'legislation'. Must be one of: {VALID_LEGISLATIONS} or empty.")
    elif isinstance(legislation, list): # Handle list if sent (though Case doc implies single)
         legislation = _validate_list_values(legislation, VALID_LEGISLATIONS, 'legislation')

    try: # Validate bools and dates
        waiting_on_client = _validate_bool(waiting_on_client)
        self_submitted = _validate_bool(self_submitted)
        paperwork_received = _validate_bool(paperwork_received)
        paperwork_recurring_reminder = _validate_bool(paperwork_recurring_reminder)
        h4h_send_initial_approval = _validate_bool(h4h_send_initial_approval)
        h4h_initial_approved = _validate_bool(h4h_initial_approved)
        consent_received_date = _validate_date(consent_received_date)
        paperwork_due_date = _validate_date(paperwork_due_date)
    except ValidationError as e: raise e

    crm_data = load_crm_data()
    if str(support_item_id) not in crm_data.get('support_items', {}): raise NotFoundError(f"Support item with ID {support_item_id} not found.")

    new_id = crm_data['next_case_id']; now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    new_case_data = { # Store validated values
        "id": new_id, "support_item_id": support_item_id, "summary": summary, "details": details,
        "case_name": case_name.strip(), "case_area": case_area, "type": case_type, "sub_type": sub_type,
        "waiting_on_client": waiting_on_client, "consent_received_date": consent_received_date,
        "claim_type": claim_type, "self_submitted": self_submitted, "legislation": legislation,
        "paperwork_due_date": paperwork_due_date, "paperwork_received": paperwork_received,
        "paperwork_recurring_reminder": paperwork_recurring_reminder, "description": description,
        "h4h_approval_team": h4h_approval_team, "h4h_send_initial_approval": h4h_send_initial_approval,
        "h4h_initial_approved": h4h_initial_approved, "h4h_approved_by": h4h_approved_by,
        "allocated_property": allocated_property, "assigned_case_manager": assigned_case_manager,
        "created_at": now_iso, "updated_at": now_iso
    }
    crm_data['cases'][str(new_id)] = new_case_data; crm_data['next_case_id'] += 1
    save_crm_data(crm_data)
    logging.info(f"Created case (JSON): ID={new_id} for Support Item ID={support_item_id}")
    return new_case_data

def get_cases_logic(filters):
    """Logic to get cases based on filters."""
    crm_data = load_crm_data()
    all_cases = list(crm_data.get('cases', {}).values())
    filtered_cases = all_cases
    if 'support_item_id' in filters:
         support_item_id = filters['support_item_id']
         if isinstance(support_item_id, int): filtered_cases = [c for c in filtered_cases if c.get('support_item_id') == support_item_id]
         else: raise ValidationError("Invalid 'support_item_id' filter format.")
    return sorted(filtered_cases, key=lambda case: case.get('created_at', ''), reverse=True)

def get_case_logic(case_id):
    """Logic to get a single case and its parent contact_id."""
    crm_data = load_crm_data()
    case_id_str = str(case_id)
    case = crm_data.get('cases', {}).get(case_id_str)
    if not case: raise NotFoundError("Case not found")

    # Find parent support item to get contact_id
    support_item_id = case.get('support_item_id')
    contact_id = None
    if support_item_id:
        support_item = crm_data.get('support_items', {}).get(str(support_item_id))
        if support_item: contact_id = support_item.get('contact_id')
        else: logging.warning(f"Case {case_id} references missing support item {support_item_id}")

    # Calculate claim item count
    claim_items = crm_data.get('claim_items', {})
    count = sum(1 for ci in claim_items.values() if ci.get('case_id') == case_id)

    case_detail = case.copy()
    case_detail["claim_item_count"] = count
    case_detail["contact_id"] = contact_id # Add parent contact ID

    return case_detail

def update_case_logic(case_id, data):
    """Logic to update a case."""
    case_id_str = str(case_id)
    crm_data = load_crm_data()
    cases = crm_data.get('cases', {})
    if case_id_str not in cases: raise NotFoundError("Case not found")

    case_to_update = cases[case_id_str]
    original_case = case_to_update.copy()

    # Define updatable fields
    str_fields = ['summary', 'details', 'legislation', 'type', 'sub_type', 'claim_type', 'description', 'case_name', 'h4h_approval_team', 'h4h_approved_by', 'allocated_property', 'assigned_case_manager', 'case_area']
    bool_fields = ['waiting_on_client', 'self_submitted', 'paperwork_received', 'paperwork_recurring_reminder', 'h4h_send_initial_approval', 'h4h_initial_approved']
    date_fields = ['consent_received_date', 'paperwork_due_date']

    for field in str_fields:
        if field in data:
            value = data[field]
            if isinstance(value, str): value = value.strip()
            # Special validation for legislation
            if field == 'legislation':
                if value and value not in VALID_LEGISLATIONS: raise ValidationError(f"Invalid 'legislation'. Must be one of: {VALID_LEGISLATIONS} or empty.")
            case_to_update[field] = value if value is not None else ''
    for field in bool_fields:
        if field in data:
            try: case_to_update[field] = _validate_bool(data[field])
            except ValidationError as e: raise ValidationError(f"Invalid boolean field '{field}': {e}")
    for field in date_fields:
        if field in data:
            try: case_to_update[field] = _validate_date(data[field])
            except ValidationError as e: raise ValidationError(f"Invalid date field '{field}': {e}")

    if case_to_update != original_case:
        case_to_update['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_crm_data(crm_data)
        logging.info(f"Updated case (JSON): ID={case_id}")
        return case_to_update
    else:
        return None # No changes

def delete_case_logic(case_id):
    """Logic to delete a case and cascade claim items."""
    case_id_str = str(case_id)
    crm_data = load_crm_data()
    if case_id_str not in crm_data.get('cases', {}): raise NotFoundError("Case not found")

    # Find and delete related claim items
    claim_items_to_delete_ids = [ci_id for ci_id, ci in crm_data.get('claim_items', {}).items() if ci.get('case_id') == case_id]
    for ci_id in claim_items_to_delete_ids:
        if ci_id in crm_data.get('claim_items', {}):
            del crm_data['claim_items'][ci_id]
            logging.debug(f"Deleting claim item {ci_id} (cascade from case {case_id})")

    # Delete the case itself
    del crm_data['cases'][case_id_str]
    logging.debug(f"Deleting case {case_id}")

    save_crm_data(crm_data)
    logging.info(f"Deleted case (JSON): ID={case_id} and cascaded claim items.")
    return True

# -- Claim Items --
def create_claim_item_logic(data):
    """Logic to create a new claim item."""
    case_id = data.get('case_id')
    contact_id = data.get('contact_id') # Now pre-filled when creating from Case view
    condition = data.get('condition')
    classification = data.get('classification')
    sop = data.get('sop')
    reasonable_hypothesis = data.get('reasonable_hypothesis')
    legislations = data.get('legislations', [])
    medical_date_identification = data.get('medical_date_identification')
    medical_date_diagnosis = data.get('medical_date_diagnosis')
    determined_by_admin = data.get('determined_by_admin')
    rejection_reason = data.get('rejection_reason')
    determined_by = data.get('determined_by')

    # Validation
    if not isinstance(case_id, int) or case_id <= 0: raise ValidationError("Missing or invalid 'case_id'")
    if not isinstance(contact_id, int) or contact_id <= 0: raise ValidationError("Missing or invalid 'contact_id'")
    if not condition or not isinstance(condition, str) or not condition.strip(): raise ValidationError("Missing or invalid 'condition'")
    legislations = _validate_list_values(legislations, VALID_LEGISLATIONS, 'legislations')
    try:
        medical_date_identification = _validate_date(medical_date_identification)
        medical_date_diagnosis = _validate_date(medical_date_diagnosis)
    except ValidationError as e: raise e

    crm_data = load_crm_data()
    if str(case_id) not in crm_data.get('cases', {}): raise NotFoundError(f"Case with ID {case_id} not found.")
    if str(contact_id) not in crm_data.get('contacts', {}): raise NotFoundError(f"Contact with ID {contact_id} not found.")

    new_id = crm_data['next_claim_item_id']; now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    new_claim_item_data = { # Store validated values
        "id": new_id, "case_id": case_id, "contact_id": contact_id, "condition": condition.strip(),
        "classification": classification, "sop": sop, "reasonable_hypothesis": reasonable_hypothesis,
        "legislations": legislations, "medical_date_identification": medical_date_identification,
        "medical_date_diagnosis": medical_date_diagnosis, "determined_by_admin": determined_by_admin,
        "rejection_reason": rejection_reason, "determined_by": determined_by,
        "created_at": now_iso, "updated_at": now_iso
    }
    crm_data['claim_items'][str(new_id)] = new_claim_item_data; crm_data['next_claim_item_id'] += 1
    save_crm_data(crm_data)
    logging.info(f"Created claim item (JSON): ID={new_id} for Case ID={case_id}")
    return new_claim_item_data

def get_claim_items_logic(filters):
    """Logic to get claim items based on filters."""
    crm_data = load_crm_data()
    all_claim_items = list(crm_data.get('claim_items', {}).values())
    filtered_items = all_claim_items
    if 'case_id' in filters:
         case_id = filters['case_id']
         if isinstance(case_id, int): filtered_items = [ci for ci in filtered_items if ci.get('case_id') == case_id]
         else: raise ValidationError("Invalid 'case_id' filter format.")
    if 'contact_id' in filters:
         contact_id = filters['contact_id']
         if isinstance(contact_id, int): filtered_items = [ci for ci in filtered_items if ci.get('contact_id') == contact_id]
         else: raise ValidationError("Invalid 'contact_id' filter format.")
    return sorted(filtered_items, key=lambda ci: ci.get('created_at', ''), reverse=True)

def get_claim_item_logic(claim_item_id):
    """Logic to get a single claim item."""
    crm_data = load_crm_data()
    claim_item = crm_data.get('claim_items', {}).get(str(claim_item_id))
    if not claim_item: raise NotFoundError("Claim item not found")
    return claim_item

def update_claim_item_logic(claim_item_id, data):
    """Logic to update a claim item."""
    claim_item_id_str = str(claim_item_id)
    crm_data = load_crm_data()
    claim_items = crm_data.get('claim_items', {})
    if claim_item_id_str not in claim_items: raise NotFoundError("Claim item not found")

    item_to_update = claim_items[claim_item_id_str]
    original_item = item_to_update.copy()

    # Update fields if present in data
    str_fields = ['condition', 'classification', 'sop', 'reasonable_hypothesis', 'determined_by_admin', 'rejection_reason', 'determined_by']
    list_fields = ['legislations']
    date_fields = ['medical_date_identification', 'medical_date_diagnosis']

    for field in str_fields:
        if field in data:
            value = data[field]; item_to_update[field] = value.strip() if isinstance(value, str) else (value if value is not None else '')
    for field in list_fields:
        if field in data:
            value = data[field]
            if field == 'legislations': value = _validate_list_values(value, VALID_LEGISLATIONS, field)
            else:
                if not isinstance(value, list): raise ValidationError(f"'{field}' must be a list")
            item_to_update[field] = value
    for field in date_fields:
        if field in data:
            try: item_to_update[field] = _validate_date(data[field])
            except ValidationError as e: raise ValidationError(f"Invalid date field '{field}': {e}")

    if item_to_update != original_item:
        item_to_update['updated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_crm_data(crm_data)
        logging.info(f"Updated claim item (JSON): ID={claim_item_id}")
        return item_to_update
    else:
        return None # No changes

def delete_claim_item_logic(claim_item_id):
    """Logic to delete a claim item."""
    claim_item_id_str = str(claim_item_id)
    crm_data = load_crm_data()
    if claim_item_id_str not in crm_data.get('claim_items', {}): raise NotFoundError("Claim item not found")
    del crm_data['claim_items'][claim_item_id_str]
    save_crm_data(crm_data)
    logging.info(f"Deleted claim item (JSON): ID={claim_item_id}")
    return True
