-- Authentication tables
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'staff', 'customer')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

CREATE TABLE user_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    session_start TIMESTAMP NOT NULL,
    session_end TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT
);

-- Inventory tables
CREATE TABLE suppliers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    contact_name VARCHAR(100),
    email VARCHAR(100),
    phone VARCHAR(20),
    address TEXT
);

CREATE TABLE inventory_items (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    category VARCHAR(50),
    sku VARCHAR(50) UNIQUE,
    barcode VARCHAR(50),
    unit_price DECIMAL(10, 2) NOT NULL,
    cost_price DECIMAL(10, 2) NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 0,
    reorder_level INTEGER,
    supplier_id INTEGER REFERENCES suppliers(id),
    storage_location VARCHAR(100),
    expiry_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insights tables
CREATE TABLE insights (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(50),
    source_data VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id),
    tags TEXT[],
    impact_score DECIMAL(5, 2),
    confidence_level DECIMAL(5, 2),
    trend_direction VARCHAR(20),
    seasonality_index DECIMAL(5, 2),
    anomaly_score DECIMAL(5, 2),
    actionability_score DECIMAL(5, 2),
    relevance_score DECIMAL(5, 2)
);

-- Analysis tables
CREATE TABLE analyses (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    data_source VARCHAR(100),
    parameters JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by INTEGER REFERENCES users(id),
    status VARCHAR(20) CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    result_summary TEXT,
    statistical_significance DECIMAL(10, 5),
    correlation_coefficients JSONB,
    regression_coefficients JSONB,
    cluster_centers JSONB,
    variance_explained DECIMAL(5, 2),
    prediction_accuracy DECIMAL(5, 2),
    execution_time INTEGER -- in seconds
);

-- Projects tables
CREATE TABLE projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    start_date DATE,
    end_date DATE,
    budget DECIMAL(12, 2),
    status VARCHAR(20),
    priority VARCHAR(20),
    manager_id INTEGER REFERENCES users(id),
    completion_percentage DECIMAL(5, 2) DEFAULT 0,
    days_remaining INTEGER,
    budget_utilization DECIMAL(5, 2) DEFAULT 0,
    burn_rate DECIMAL(10, 2),
    risk_score DECIMAL(5, 2),
    on_track_status BOOLEAN DEFAULT TRUE,
    efficiency_score DECIMAL(5, 2)
);

CREATE TABLE project_team_members (
    project_id INTEGER REFERENCES projects(id),
    user_id INTEGER REFERENCES users(id),
    PRIMARY KEY (project_id, user_id)
);

CREATE TABLE project_milestones (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    title VARCHAR(100) NOT NULL,
    description TEXT,
    due_date DATE,
    completed BOOLEAN DEFAULT FALSE
);

-- Staff tables
CREATE TABLE staff (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    position VARCHAR(100),
    department VARCHAR(50),
    hire_date DATE,
    salary DECIMAL(10, 2),
    manager_id INTEGER REFERENCES staff(id),
    is_active BOOLEAN DEFAULT TRUE,
    tenure INTEGER, -- in days
    productivity_score DECIMAL(5, 2),
    utilization_rate DECIMAL(5, 2),
    efficiency_rating DECIMAL(5, 2),
    performance_index DECIMAL(5, 2),
    revenue_contribution DECIMAL(12, 2),
    team_effectiveness DECIMAL(5, 2),
    training_completion DECIMAL(5, 2)
);

-- Customers tables
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    address TEXT,
    signup_date DATE DEFAULT CURRENT_DATE,
    last_purchase_date DATE,
    segment VARCHAR(50),
    preferences JSONB,
    marketing_opt_in BOOLEAN DEFAULT TRUE,
    lifetime_value DECIMAL(12, 2) DEFAULT 0,
    churn_risk DECIMAL(5, 2),
    days_since_last_purchase INTEGER,
    average_order_value DECIMAL(10, 2),
    purchase_frequency DECIMAL(5, 2),
    loyalty_score DECIMAL(5, 2),
    satisfaction_index DECIMAL(5, 2),
    engagement_score DECIMAL(5, 2)
);

-- Analytics tables
CREATE TABLE analytics_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    user_id INTEGER REFERENCES users(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(100),
    session_id VARCHAR(100),
    page_url TEXT,
    device_type VARCHAR(50),
    browser VARCHAR(50),
    location VARCHAR(100)
);

CREATE TABLE analytics_metrics (
    id SERIAL PRIMARY KEY,
    date DATE DEFAULT CURRENT_DATE,
    user_retention DECIMAL(5, 2),
    conversion_rate DECIMAL(5, 2),
    bounce_rate DECIMAL(5, 2),
    average_session_duration INTEGER, -- in seconds
    page_load_time INTEGER, -- in milliseconds
    user_flow_pattern JSONB,
    feature_usage_frequency JSONB,
    engagement_depth DECIMAL(5, 2),
    funnel_completion_rate DECIMAL(5, 2),
    user_growth_rate DECIMAL(5, 2)
);

-- Database migration tables
CREATE TABLE migrations (
    id SERIAL PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_at TIMESTAMP,
    script_path VARCHAR(255),
    is_applied BOOLEAN DEFAULT FALSE,
    schema_health_score DECIMAL(5, 2),
    migration_success_rate DECIMAL(5, 2)
);

CREATE TABLE database_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    db_size BIGINT, -- in bytes
    table_count INTEGER,
    index_coverage DECIMAL(5, 2),
    query_performance_index DECIMAL(5, 2),
    index_fragmentation DECIMAL(5, 2),
    connection_utilization DECIMAL(5, 2)
);
-- Update restaurants table to match the importer's expectations
CREATE TABLE restaurants (
    id SERIAL PRIMARY KEY,
    public_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    cuisine_type VARCHAR(100),
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
);

-- Add an error logging table to track import issues
CREATE TABLE import_logs (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(100) NOT NULL,
    error_message TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Update restaurants table to include separate price range columns
ALTER TABLE restaurants 
ADD COLUMN IF NOT EXISTS price_min INTEGER,
ADD COLUMN IF NOT EXISTS price_max INTEGER;

-- Add calculated fields to restaurants table
ALTER TABLE restaurants 
ADD COLUMN IF NOT EXISTS avg_review_score DECIMAL(3,2),
ADD COLUMN IF NOT EXISTS review_count INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS popularity_score DECIMAL(5,2),
ADD COLUMN IF NOT EXISTS days_since_update INTEGER,
ADD COLUMN IF NOT EXISTS price_category VARCHAR(20),
ADD COLUMN IF NOT EXISTS distance_from_center DECIMAL(10,2),
ADD COLUMN IF NOT EXISTS cuisine_type VARCHAR(50),
ADD COLUMN IF NOT EXISTS is_premium BOOLEAN DEFAULT FALSE;

-- Create a function to update calculated fields
CREATE OR REPLACE FUNCTION update_restaurant_calculated_fields()
RETURNS TRIGGER AS $$
BEGIN
    -- Calculate days since update
    NEW.days_since_update := EXTRACT(DAY FROM NOW() - NEW.updated_at);
    
    -- Determine price category based on price_min and price_max
    IF NEW.price_max IS NOT NULL THEN
        CASE 
            WHEN NEW.price_max < 200 THEN NEW.price_category := 'Budget';
            WHEN NEW.price_max BETWEEN 200 AND 500 THEN NEW.price_category := 'Moderate';
            WHEN NEW.price_max BETWEEN 501 AND 1000 THEN NEW.price_category := 'Upscale';
            WHEN NEW.price_max > 1000 THEN NEW.price_category := 'Fine Dining';
            ELSE NEW.price_category := 'Unknown';
        END CASE;
    END IF;
    
    -- Set is_premium based on price category and rating
    IF (NEW.price_category IN ('Upscale', 'Fine Dining') AND NEW.rating >= 4.0) THEN
        NEW.is_premium := TRUE;
    ELSE
        NEW.is_premium := FALSE;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update calculated fields
CREATE TRIGGER update_restaurant_calculated_fields_trigger
BEFORE INSERT OR UPDATE ON restaurants
FOR EACH ROW
EXECUTE FUNCTION update_restaurant_calculated_fields();

-- Add raw_json_data table for debugging imports
CREATE TABLE raw_json_data (
    id SERIAL PRIMARY KEY,
    file_name VARCHAR(255) NOT NULL,
    json_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
