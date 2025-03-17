import json
import os
import pandas as pd
from pathlib import Path
import psycopg2
from psycopg2.extras import execute_values
import logging
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection parameters - update these with your actual database credentials
DB_PARAMS = {
    "dbname": "bitebasedb",
    "user": "bitebase_admin",
    "password": "npg_U1lvFbx4egOr",
    "host": "ep-broad-lake-a5n1vqt1.us-east-2.aws.neon.tech",  # Fixed hostname format
    "port": "5432",
    "sslmode": "require"  # Added SSL mode for cloud database
}

def connect_to_db():
    """Establish connection to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        logger.info("Connected to database successfully")
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise

def map_restaurant_to_db_schema(restaurant):
    """
    Map restaurant JSON data to database schema fields
    Returns a dictionary with mapped fields
    """
    # Basic restaurant info mapping
    mapped_data = {
        "name": restaurant.get("name", "")[:50] if restaurant.get("name") else "",
        "description": restaurant.get("description", ""),
        "category": "",  # Default empty string
        "phone": restaurant.get("phoneNumber", "")[:50] if restaurant.get("phoneNumber") else "",
        "email": restaurant.get("email", "")[:50] if restaurant.get("email") else "",
        "website": restaurant.get("website", "")[:50] if restaurant.get("website") else "",
        "public_id": restaurant.get("publicId", "")[:50] if restaurant.get("publicId") else "",
        "rating": float(restaurant.get("rating", 0)) if not isinstance(restaurant.get("rating"), dict) else 0.0,
        "is_active": True,  # Default value
        "created_at": "NOW()",  # Use database function for current timestamp
        "price_min": None,
        "price_max": None,
        "price_range": ""
    }
    
    # Handle reviews data if available
    reviews = restaurant.get("reviews", [])
    if isinstance(reviews, list) and len(reviews) > 0:
        # Calculate average review score
        total_score = sum(review.get("rating", 0) for review in reviews if "rating" in review)
        mapped_data["avg_review_score"] = round(total_score / len(reviews), 2) if len(reviews) > 0 else None
        mapped_data["review_count"] = len(reviews)
    
    # Extract cuisine type from categories if available
    categories = restaurant.get("categories", [])
    if isinstance(categories, list) and len(categories) > 0:
        # Use the first category as the primary cuisine type
        mapped_data["cuisine_type"] = categories[0].get("name", "") if isinstance(categories[0], dict) else str(categories[0])
        # Use all categories as the general category
        mapped_data["category"] = ", ".join(cat.get("name", "") if isinstance(cat, dict) else str(cat) for cat in categories)
    
    # Calculate popularity score (simple algorithm based on rating and review count)
    if mapped_data.get("rating") and mapped_data.get("review_count"):
        mapped_data["popularity_score"] = round(float(mapped_data["rating"]) * math.log(1 + mapped_data.get("review_count", 0)), 2)
    
    # Handle address from contact.address.street if available
    contact = restaurant.get("contact", {})
    if isinstance(contact, dict) and isinstance(contact.get("address"), dict):
        street = contact["address"].get("street", "")
        hint = contact["address"].get("hint", "")
        
        # Get district, subdistrict and city if available
        district = contact["address"].get("district", {}).get("name", "") if isinstance(contact["address"].get("district"), dict) else ""
        subdistrict = contact["address"].get("subDistrict", {}).get("name", "") if isinstance(contact["address"].get("subDistrict"), dict) else ""
        city = contact["address"].get("city", {}).get("name", "") if isinstance(contact["address"].get("city"), dict) else ""
        
        # Combine address components
        full_address = street
        if hint:
            full_address += f" ({hint})"
        if subdistrict or district or city:
            address_parts = [p for p in [subdistrict, district, city] if p]
            if address_parts:
                full_address += f", {', '.join(address_parts)}"
        
        mapped_data["address"] = full_address
    elif isinstance(restaurant.get("address"), dict) and restaurant["address"].get("streetAddress"):
        # Fallback to old address format
        mapped_data["address"] = restaurant["address"].get("streetAddress", "")
    else:
        mapped_data["address"] = ""
    
    # Handle price_range as a dictionary with min and max values
    price_range = restaurant.get("priceRange", restaurant.get("price_range"))
    if isinstance(price_range, dict):
        # Store the formatted price range (e.g., "$$ - $$$")
        mapped_data["price_range"] = price_range.get("name", "")
        
        # Store min and max values in separate columns
        if "min" in price_range and "max" in price_range:
            mapped_data["price_min"] = price_range.get("min")
            mapped_data["price_max"] = price_range.get("max")
            mapped_data["price_range"] = f"{price_range.get('min')} - {price_range.get('max')}"
        elif "value" in price_range:
            # Some sources use a single value instead of min/max
            mapped_data["price_min"] = price_range.get("value")
            mapped_data["price_max"] = price_range.get("value")
            mapped_data["price_range"] = str(price_range.get("value", ""))
    else:
        # Handle case where price_range is a string or other type
        mapped_data["price_range"] = str(price_range) if price_range else ""
    
    # Extract zipcode if available
    if isinstance(contact, dict) and isinstance(contact.get("address"), dict):
        mapped_data["zipcode"] = contact["address"].get("zipCode", "")
    elif "zipcode" in restaurant:
        # Direct zipcode field in the root object
        mapped_data["zipcode"] = restaurant.get("zipcode", "")
    else:
        mapped_data["zipcode"] = ""
    
    # Extract location data if available
    if "lat" in restaurant and "lng" in restaurant:
        mapped_data["latitude"] = float(restaurant["lat"])
        mapped_data["longitude"] = float(restaurant["lng"])
    else:
        mapped_data["latitude"] = None
        mapped_data["longitude"] = None
    
    # Extract operating hours if available
    if "hours" in restaurant and restaurant["hours"]:
        try:
            # Convert dict to JSON string
            mapped_data["operating_hours"] = json.dumps(restaurant["hours"])
        except:
            mapped_data["operating_hours"] = None
    else:
        mapped_data["operating_hours"] = None
    
    # Extract menu items if available
    if "menu" in restaurant and isinstance(restaurant["menu"], list):
        try:
            # Convert list to JSON string
            mapped_data["menu_items"] = json.dumps(restaurant["menu"])
        except:
            mapped_data["menu_items"] = None
    else:
        mapped_data["menu_items"] = None
    
    return mapped_data

def create_restaurants_table(conn):
    """Create restaurants table if it doesn't exist"""
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS restaurants (
            id SERIAL PRIMARY KEY,
            public_id VARCHAR(100) UNIQUE NOT NULL,
            name VARCHAR(200) NOT NULL,
            description TEXT,
            category VARCHAR(100),
            address TEXT,
            zipcode VARCHAR(20),
            phone VARCHAR(50),
            email VARCHAR(100),
            website VARCHAR(255),
            rating DECIMAL(3,1),
            price_range VARCHAR(50),
            price_min DECIMAL(10,2),
            price_max DECIMAL(10,2),
            latitude DECIMAL(10,7),
            longitude DECIMAL(10,7),
            operating_hours JSONB,
            menu_items JSONB,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        logger.info("Restaurants table created or already exists")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating restaurants table: {e}")
        raise
    finally:
        cursor.close()

def alter_restaurants_table(conn):
    """Add any missing columns to the restaurants table"""
    cursor = conn.cursor()
    try:
        # Check if cuisine_type column exists
        cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='restaurants' AND column_name='cuisine_type'
        """)
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE restaurants ADD COLUMN cuisine_type VARCHAR(100)")
            logger.info("Added cuisine_type column to restaurants table")
        
        conn.commit()
        logger.info("Restaurants table altered successfully")
    except Exception as e:
        logger.error(f"Error altering restaurants table: {e}")
        conn.rollback()
        raise

def insert_restaurant_data(conn, restaurants_data):
    """Insert mapped restaurant data into database"""
    if not restaurants_data:
        logger.warning("No restaurant data to insert")
        return
    
    cursor = conn.cursor()
    try:
        # Prepare column names and values for insertion
        columns = restaurants_data[0].keys()
        column_str = ", ".join(columns)
        
        # Debug: Check each value's type
        for restaurant in restaurants_data:
            for col, val in restaurant.items():
                if isinstance(val, dict) or isinstance(val, list):
                    logger.error(f"Column {col} contains a {type(val).__name__}: {val}")
                    # Convert to JSON string if it's a dict or list
                    restaurant[col] = json.dumps(val) if val is not None else None
        
        value_placeholders = ", ".join(["%s"] * len(columns))
        
        # Prepare values list
        values = [[restaurant[col] for col in columns] for restaurant in restaurants_data]
        
        # Create SQL query with ON CONFLICT clause to handle duplicates
        query = f"""
        INSERT INTO restaurants ({column_str})
        VALUES %s
        ON CONFLICT (public_id) 
        DO UPDATE SET 
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            category = EXCLUDED.category,
            address = EXCLUDED.address,
            zipcode = EXCLUDED.zipcode,
            phone = EXCLUDED.phone,
            email = EXCLUDED.email,
            website = EXCLUDED.website,
            rating = EXCLUDED.rating,
            price_range = EXCLUDED.price_range,
            price_min = EXCLUDED.price_min,
            price_max = EXCLUDED.price_max,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            operating_hours = EXCLUDED.operating_hours,
            menu_items = EXCLUDED.menu_items,
            updated_at = NOW()
        """
        
        # Execute the query with all values
        execute_values(cursor, query, values)
        conn.commit()
        logger.info(f"Successfully inserted {len(restaurants_data)} restaurants")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error inserting restaurant data: {e}")
        raise
    finally:
        cursor.close()

def import_restaurants_from_file(file_path, conn=None, progress=None, task_id=None):
    """Import restaurants from a JSON file into the database"""
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Load JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Store raw JSON for debugging if we have a connection
        if conn:
            try:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO raw_json_data (file_name, json_data) VALUES (%s, %s)",
                        (os.path.basename(file_path), json.dumps(data))
                    )
                    conn.commit()
            except psycopg2.errors.UndefinedTable:
                # Create the table if it doesn't exist
                conn.rollback()  # Roll back the failed transaction
                with conn.cursor() as cursor:
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS raw_json_data (
                        id SERIAL PRIMARY KEY,
                        file_name VARCHAR(255) NOT NULL,
                        json_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """)
                    conn.commit()
                    # Try inserting again
                    cursor.execute(
                        "INSERT INTO raw_json_data (file_name, json_data) VALUES (%s, %s)",
                        (os.path.basename(file_path), json.dumps(data))
                    )
                    conn.commit()
                    logger.info("Created raw_json_data table and inserted data")
        
        # Extract restaurants list from the JSON structure - use the same approach as restaurant_parser.py
        restaurants = []
        if isinstance(data, dict) and 'page' in data and 'entities' in data['page']:
            restaurants = data['page']['entities']
            logger.info(f"Found restaurants at path: page -> entities")
        
        # If no restaurants found, try other paths
        if not restaurants and isinstance(data, dict):
            # Try other common paths
            paths_to_try = [
                ('page', 'searchResult', 'entities'),
                ('restaurants',),
                ('data', 'restaurants'),
                ('entities',),
                ('results',),
                ('page', 'searchResults'),
                ('page', 'restaurants')
            ]
            
            for path in paths_to_try:
                current = data
                valid_path = True
                for key in path:
                    if isinstance(current, dict) and key in current:
                        current = current[key]
                    else:
                        valid_path = False
                        break
                
                if valid_path and isinstance(current, list):
                    restaurants = current
                    logger.info(f"Found restaurants at path: {' -> '.join(path)}")
                    break
        
        if not restaurants:
            # Log the structure to debug
            structure = list(data.keys()) if isinstance(data, dict) else type(data)
            logger.warning(f"No restaurants found in {file_path}. JSON structure: {structure}")
            
            # Save a sample of the raw JSON to a debug file for inspection
            debug_dir = "debug_json"
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = os.path.join(debug_dir, f"debug_{os.path.basename(file_path)}")
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved debug JSON to {debug_file}")
            
            # Record the error in the import_logs table
            if conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO import_logs (file_name, error_message) VALUES (%s, %s)",
                        (os.path.basename(file_path), f"No restaurants found. Structure: {structure}")
                    )
                    conn.commit()
            
            if progress and task_id:
                progress.update(task_id, completed=1.0)
            return 0
        
        logger.info(f"Found {len(restaurants)} restaurants in {file_path}")
        
        # Map restaurants to database schema
        mapped_restaurants = []
        skipped_count = 0
        for restaurant in restaurants:
            mapped_data = map_restaurant_to_db_schema(restaurant)
            # Only add if we have required fields
            if mapped_data.get('name') and mapped_data.get('public_id'):
                mapped_restaurants.append(mapped_data)
            else:
                skipped_count += 1
                logger.warning(f"Skipping restaurant with missing required fields: {restaurant.get('name', 'Unknown')}")
        
        logger.info(f"Mapped {len(mapped_restaurants)} restaurants, skipped {skipped_count}")
        
        # Insert into database
        if conn and mapped_restaurants:
            inserted_count = insert_restaurant_data(conn, mapped_restaurants)
            logger.info(f"Inserted {inserted_count} restaurants from {file_path}")
            return inserted_count
        elif not conn:
            logger.warning("No database connection provided, skipping insert")
            return 0
        else:
            logger.warning("No valid restaurants to insert")
            return 0
            
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        # Record the error in the import_logs table
        if conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO import_logs (file_name, error_message) VALUES (%s, %s)",
                    (os.path.basename(file_path), str(e))
                )
                conn.commit()
        if progress and task_id:
            progress.update(task_id, completed=1.0)
        return 0

def process_all_files(input_dir, file_pattern="*.json", resume=False):
    """Process all JSON files in the directory and store in database"""
    # Get all JSON files in the input directory
    json_files = list(Path(input_dir).glob(file_pattern))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Connect to database
    conn = connect_to_db()
    
    # Create state tracking table
    create_processing_state_table(conn)
    
    # Get already processed files if resuming
    processed_files = []
    if resume:
        processed_files = get_processed_files(conn)
        logger.info(f"Resuming processing. {len(processed_files)} files already processed.")
    
    # Filter out already processed files if resuming
    if resume:
        json_files = [f for f in json_files if f.name not in processed_files]
        logger.info(f"{len(json_files)} files remaining to process.")
    
    # Initialize counters
    total_restaurants_processed = 0
    total_files_processed = 0
    
    try:
        # Create progress bar
        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
        )
        
        with progress:
            # Create a task for overall progress
            overall_task = progress.add_task(f"Processing {len(json_files)} files", total=len(json_files))
            
            for json_file in json_files:
                try:
                    # Create a task for this file
                    file_task = progress.add_task(f"Processing {json_file.name}", total=1.0)
                    
                    # Process the file
                    restaurants_count = import_restaurants_from_file(str(json_file), conn, progress, file_task)
                    
                    # Mark file as processed
                    mark_file_as_processed(conn, json_file.name, restaurants_count)
                    
                    # Update counters
                    total_restaurants_processed += restaurants_count
                    total_files_processed += 1
                    
                    # Update overall progress
                    progress.update(overall_task, completed=total_files_processed)
                    
                    # Log progress periodically
                    if total_files_processed % 5 == 0 or total_files_processed == len(json_files):
                        logger.info(f"Progress: {total_files_processed}/{len(json_files)} files, {total_restaurants_processed} restaurants processed")
                        
                except Exception as e:
                    logger.error(f"Error processing file {json_file}: {e}")
                    # Continue with next file instead of stopping
                    continue
    
    except KeyboardInterrupt:
        logger.info(f"Process interrupted by user. Processed {total_files_processed}/{len(json_files)} files and {total_restaurants_processed} restaurants.")
    except Exception as e:
        logger.error(f"Error processing files: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")
        
        # Print final summary
        logger.info(f"Summary: Processed {total_files_processed}/{len(json_files)} files and {total_restaurants_processed} restaurants.")
        
    return total_restaurants_processed

def clear_restaurants_table(conn):
    """Clear all data from the restaurants table"""
    cursor = conn.cursor()
    try:
        cursor.execute("TRUNCATE TABLE restaurants RESTART IDENTITY")
        conn.commit()
        logger.info("Restaurants table cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing restaurants table: {e}")
        conn.rollback()
        raise

def create_processing_state_table(conn):
    """Create a table to track processing state for resuming"""
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_state (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255) UNIQUE NOT NULL,
            processed BOOLEAN DEFAULT FALSE,
            processed_at TIMESTAMP,
            restaurants_count INTEGER DEFAULT 0
        )
        """)
        conn.commit()
        logger.info("Processing state table created or already exists")
    except Exception as e:
        logger.error(f"Error creating processing state table: {e}")
        conn.rollback()
        raise

def mark_file_as_processed(conn, file_name, restaurants_count):
    """Mark a file as processed in the state table"""
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO processing_state (file_name, processed, processed_at, restaurants_count)
        VALUES (%s, TRUE, NOW(), %s)
        ON CONFLICT (file_name) 
        DO UPDATE SET 
            processed = TRUE,
            processed_at = NOW(),
            restaurants_count = %s
        """, (file_name, restaurants_count, restaurants_count))
        conn.commit()
    except Exception as e:
        logger.error(f"Error marking file as processed: {e}")
        conn.rollback()
        raise

def get_processed_files(conn):
    """Get a list of already processed files"""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT file_name FROM processing_state WHERE processed = TRUE")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error getting processed files: {e}")
        return []

def create_raw_json_data_table(conn):
    """Create raw_json_data table if it doesn't exist"""
    cursor = conn.cursor()
    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_json_data (
            id SERIAL PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            json_data JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()
        logger.info("Raw JSON data table created or already exists")
    except Exception as e:
        logger.error(f"Error creating raw_json_data table: {e}")
        conn.rollback()
        raise

if __name__ == "__main__":
    import argparse
    import signal
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Import restaurant data to database')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing restaurant JSON files')
    parser.add_argument('--clear-table', action='store_true', help='Clear restaurants table before importing')
    parser.add_argument('--resume', action='store_true', help='Resume processing from where it left off')
    
    args = parser.parse_args()
    
    # Connect to database
    conn = connect_to_db()
    
    try:
        # Create tables if they don't exist
        create_restaurants_table(conn)
        alter_restaurants_table(conn)
        create_processing_state_table(conn)
        create_raw_json_data_table(conn)  # Add this line
        
        # Clear existing data if requested
        if args.clear_table and not args.resume:
            clear_restaurants_table(conn)
        
        # Process all files
        total_processed = process_all_files(args.data_dir, resume=args.resume)
        
        logger.info(f"Import completed successfully. Total restaurants processed: {total_processed}")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user. You can resume later with --resume flag.")
    except Exception as e:
        logger.error(f"Error in main process: {e}")
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")
