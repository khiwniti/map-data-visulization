// This file is now delivery-menu.ts

import fs from 'fs';
import { parse } from 'csv-parse';

interface RestaurantData {
  Place_ID: string;
  Location: string;
  Customer_Density: string;
  Competitor_Count: string;
  Average_Rent: string;
  Sales: string;
  Profit: string;
  Number_of_Orders: string;
  Customer_Ratings: string;
  Inventory_Turnover_Rate: string;
  Ingredient_Costs: string;
  Sales_Prices: string;
  Overhead_Costs: string;
  Gross_Margin: string;
  Net_Profit: string;
  COGS: string;
  Seasonal_Sales_Patterns: string;
  Trend_Growth_Rates: string;
  Seasonal_Ingredient_Costs: string;
  Price_Elasticity_of_Demand: string;
  Competitor_Price_Comparison: string;
  Optimal_Pricing_Points: string;
  Customer_Addresses: string;
  GPS_Data: string;
  Demographic_Data: string;
  Customer_Visit_Frequency: string;
  Market_Share_Percentage: string;
  Competitive_Advantage_Analysis: string;
  Order_Density_by_Location: string;
  Average_Delivery_Time: string;
  Customer_Satisfaction_Scores: string;
  Rental_Cost_per_Square_Foot: string;
  Property_Value_Trends: string;
  Sales_Performance_by_Location: string;
  Projected_Sales_Growth: string;
  Revenue_Forecasts: string;
  Market_Trend_Analysis: string;
  Sales_by_Day_and_Hour: string;
  Peak_Traffic_Times: string;
  Event_Impact_on_Sales: string;
  Sales_Lift_during_Promotions: string;
  Customer_Acquisition_Cost: string;
  ROI_of_Promotions: string;
  Average_Transaction_Value: string;
  Customer_Lifetime_Value: string;
  Spending_Patterns_and_Trends: string;
}

const csvFilePath = './mock_restaurant_data.csv';

fs.readFile(csvFilePath, { encoding: 'utf8' }, (err, data) => {
  if (err) {
    console.error('Failed to read CSV file:', err);
    return;
  }

  parse(data, {
    header: true,
    skipEmptyLines: true,
    columns: true,
  }, (err, records: RestaurantData[]) => {
    if (err) {
      console.error('Failed to parse CSV data:', err);
      return;
    }

    // Now you have your restaurant data in the 'records' variable
    console.log('Restaurant data:', records);

    // Example: Accessing the publicId (assuming Place_ID is similar)
    // records.forEach(record => {
    //   console.log('Public ID:', record.Place_ID);
    // });
  });
});
