Target Data Dict
= Data Dictionary

== Authentication (auth/)

=== Raw Data
- `user_id`: Unique identifier for each user
- `username`: Username for login
- `email`: User's email address
- `password_hash`: Hashed password
- `role`: User role (admin, staff, customer)
- `is_active`: Boolean indicating if account is active
- `created_at`: Timestamp of account creation
- `last_login`: Timestamp of last login

=== Calculated Data
- `session_duration`: Length of user sessions
- `login_frequency`: How often a user logs in
- `password_age`: Days since last password change
- `is_password_expired`: Boolean indicating if password needs reset

== Inventory (inventory/)

=== Raw Data
- `item_id`: Unique identifier for inventory item
- `name`: Item name
- `description`: Item description
- `category`: Item category
- `sku`: Stock keeping unit
- `barcode`: Barcode/UPC
- `unit_price`: Price per unit
- `cost_price`: Cost to acquire
- `quantity`: Current stock quantity
- `reorder_level`: Quantity triggering reorder
- `supplier_id`: ID of supplier
- `storage_location`: Where item is stored
- `expiry_date`: Expiration date for perishables
- `created_at`: When item was added to inventory
- `updated_at`: When item was last updated

=== Calculated Data
- `stock_value`: Total value of current stock (quantity × unit_price)
- `profit_margin`: Percentage profit margin ((unit_price - cost_price) / unit_price)
- `days_to_expiry`: Days until item expires
- `inventory_turnover`: Rate at which inventory is sold
- `stock_status`: Calculated status (in stock, low, out of stock)
- `days_out_of_stock`: Number of days item has been out of stock
- `reorder_needed`: Boolean indicating if reorder is needed

== Insights (insights/)

=== Raw Data
- `insight_id`: Unique identifier for insight
- `title`: Insight title
- `description`: Detailed description of insight
- `category`: Category of insight
- `source_data`: Source of the data
- `created_at`: When insight was created
- `created_by`: User who created the insight
- `tags`: Tags/keywords for the insight

=== Calculated Data
- `impact_score`: Estimated business impact score
- `confidence_level`: Statistical confidence level
- `trend_direction`: Direction of observed trend
- `seasonality_index`: Measure of seasonal patterns
- `anomaly_score`: How unusual the insight is
- `actionability_score`: How actionable the insight is
- `relevance_score`: How relevant to business goals

== Analysis (analysis/)

=== Raw Data
- `analysis_id`: Unique identifier for analysis
- `name`: Analysis name
- `description`: Analysis description
- `data_source`: Source of data being analyzed
- `parameters`: Parameters used for analysis
- `created_at`: When analysis was created
- `created_by`: User who created the analysis
- `status`: Status of analysis (pending, running, completed, failed)

=== Calculated Data
- `result_summary`: Summary of analysis results
- `statistical_significance`: P-value or other significance measure
- `correlation_coefficients`: Correlation between variables
- `regression_coefficients`: Coefficients from regression analysis
- `cluster_centers`: Centers of identified clusters
- `variance_explained`: Percentage of variance explained
- `prediction_accuracy`: Accuracy of predictive models
- `execution_time`: Time taken to run analysis

== Projects (projects/)

=== Raw Data
- `project_id`: Unique identifier for project
- `name`: Project name
- `description`: Project description
- `start_date`: Project start date
- `end_date`: Project end date
- `budget`: Allocated budget
- `status`: Project status
- `priority`: Project priority
- `manager_id`: ID of project manager
- `team_members`: List of team members
- `milestones`: Key project milestones

=== Calculated Data
- `completion_percentage`: Percentage of project completed
- `days_remaining`: Days until deadline
- `budget_utilization`: Percentage of budget used
- `burn_rate`: Rate at which budget is being used
- `resource_allocation`: Distribution of resources
- `risk_score`: Calculated risk assessment
- `on_track_status`: Boolean indicating if project is on track
- `efficiency_score`: Measure of project efficiency

== Staff (staff/)

=== Raw Data
- `staff_id`: Unique identifier for staff member
- `first_name`: First name
- `last_name`: Last name
- `email`: Email address
- `phone`: Phone number
- `position`: Job position/title
- `department`: Department
- `hire_date`: Date of hiring
- `salary`: Salary amount
- `manager_id`: ID of their manager
- `is_active`: Employment status

=== Calculated Data
- `tenure`: Length of employment in days/years
- `productivity_score`: Calculated productivity metric
- `utilization_rate`: Percentage of time utilized
- `efficiency_rating`: Efficiency rating
- `performance_index`: Overall performance index
- `revenue_contribution`: Estimated revenue contribution
- `team_effectiveness`: Team collaboration effectiveness
- `training_completion`: Percentage of required training completed

== Customers (customers/)

=== Raw Data
- `customer_id`: Unique identifier for customer
- `first_name`: First name
- `last_name`: Last name
- `email`: Email address
- `phone`: Phone number
- `address`: Physical address
- `signup_date`: Date customer was added
- `last_purchase_date`: Date of last purchase
- `segment`: Customer segment/category
- `preferences`: Customer preferences
- `marketing_opt_in`: Marketing consent status

=== Calculated Data
- `lifetime_value`: Customer lifetime value
- `churn_risk`: Probability of customer churn
- `days_since_last_purchase`: Days since last transaction
- `average_order_value`: Average value of orders
- `purchase_frequency`: How often customer makes purchases
- `loyalty_score`: Calculated loyalty metric
- `satisfaction_index`: Calculated satisfaction level
- `engagement_score`: Level of customer engagement

== Analytics (analytics/)

=== Raw Data
- `event_id`: Unique identifier for analytics event
- `event_type`: Type of event
- `user_id`: User who triggered event
- `timestamp`: When event occurred
- `source`: Source of event
- `session_id`: Session identifier
- `page_url`: URL where event occurred
- `device_type`: Type of device used
- `browser`: Browser used
- `location`: Geographic location

=== Calculated Data
- `user_retention`: User retention rate
- `conversion_rate`: Conversion rate
- `bounce_rate`: Bounce rate
- `average_session_duration`: Average session length
- `page_load_time`: Average page load time
- `user_flow_pattern`: Common navigation patterns
- `feature_usage_frequency`: How often features are used
- `engagement_depth`: Depth of user engagement
- `funnel_completion_rate`: Rate of funnel completion
- `user_growth_rate`: Rate of new user acquisition

== Init-DB (init-db/)

=== Raw Data
- `migration_id`: Unique identifier for migration
- `version`: Schema version
- `description`: Migration description
- `created_at`: When migration was created
- `applied_at`: When migration was applied
- `script_path`: Path to migration script
- `is_applied`: Whether migration has been applied

=== Calculated Data
- `schema_health_score`: Health score of database schema
- `migration_success_rate`: Success rate of migrations
- `db_size`: Current database size
- `table_count`: Number of tables
- `index_coverage`: Percentage of fields with indexes
- `query_performance_index`: Performance index of common queries
- `index_fragmentation`: Level of index fragmentation
- `connection_utilization`: Database connection utilization

