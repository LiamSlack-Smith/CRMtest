# models.py
import enum
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime, Enum as SQLEnum
from sqlalchemy.orm import relationship, sessionmaker, declarative_base
from sqlalchemy.sql import func # For default timestamps

# Define the base class for declarative models
Base = declarative_base()

# Define Enums for specific fields
class SupportItemCategory(enum.Enum):
    HOUSING = "housing"
    WELLBEING = "wellbeing"
    CLAIMS = "claims"
    EMPLOYMENT = "employment"
    OTHER = "other"

# --- Model Definitions ---

class Contact(Base):
    __tablename__ = 'contacts'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    email = Column(String(200), unique=True, index=True, nullable=True) # Allow null email initially?
    phone = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship: One Contact has many SupportItems
    # cascade="all, delete-orphan" means if a Contact is deleted, its SupportItems are also deleted.
    support_items = relationship("SupportItem", back_populates="contact", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Contact(id={self.id}, name='{self.name}')>"

class SupportItem(Base):
    __tablename__ = 'support_items'

    id = Column(Integer, primary_key=True, index=True)
    contact_id = Column(Integer, ForeignKey('contacts.id', ondelete='CASCADE'), nullable=False, index=True) # Cascade delete on FK
    category = Column(SQLEnum(SupportItemCategory, name="support_item_category_enum"), nullable=False, index=True) # Use SQLEnum
    status = Column(String(50), default="Open", index=True) # e.g., Open, In Progress, Closed
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship: Belongs to one Contact
    contact = relationship("Contact", back_populates="support_items")
    # Relationship: One SupportItem has many Cases
    cases = relationship("Case", back_populates="support_item", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SupportItem(id={self.id}, category='{self.category.value}', contact_id={self.contact_id})>"

class Case(Base):
    __tablename__ = 'cases'

    id = Column(Integer, primary_key=True, index=True)
    support_item_id = Column(Integer, ForeignKey('support_items.id', ondelete='CASCADE'), nullable=False, index=True) # Cascade delete on FK
    summary = Column(String(255), nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationship: Belongs to one SupportItem
    support_item = relationship("SupportItem", back_populates="cases")

    def __repr__(self):
        return f"<Case(id={self.id}, summary='{self.summary[:30]}...', support_item_id={self.support_item_id})>"

# --- Database Setup Function (Optional but good practice) ---
# You would call this function once, perhaps from your main script or a setup script
def create_tables(engine):
    """Creates all tables defined in the Base metadata."""
    Base.metadata.create_all(bind=engine)
    logging.info("Database tables checked/created.")

# Example usage (don't run this directly here, run from app.py or setup)
# if __name__ == "__main__":
#     # Replace with your actual database URL
#     DATABASE_URL_EXAMPLE = "postgresql://your_user:your_password@localhost:5432/crm_trainer_db"
#     engine = create_engine(DATABASE_URL_EXAMPLE)
#     create_tables(engine)
