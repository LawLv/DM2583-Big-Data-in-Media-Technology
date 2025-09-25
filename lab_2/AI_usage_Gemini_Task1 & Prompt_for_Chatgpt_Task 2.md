# Q1
I have the output from df.info() and df.isnull().sum() for my wine dataset. Based on this, act as a senior data analyst and tell me the top 3 most critical data quality issues I need to address for my analysis. For each issue, explain why it is critical.

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 432659 entries, 0 to 432658
Data columns (total 13 columns):
 No   Column              Non-Null Count   Dtype  
 0   Name                430834 non-null  object 
 1   Variety             421041 non-null  object 
 2   Origin              421017 non-null  object 
 3   Attr_1              427541 non-null  object 
 4   Attr_2              78506 non-null   object 
 5   Alcohol_vol         430767 non-null  float64
 6   Alcohol_percentage  430767 non-null  float64
 7   Winemaker_notes     98427 non-null   object 
 8   Review              95748 non-null   object 
 9   Avg_rating          46547 non-null   float64
 10  N_ratings           46547 non-null   float64
 11  Price_Out-of-stock  430766 non-null  object 
 12  Price               91226 non-null   object 
dtypes: float64(4), object(9)
memory usage: 42.9+ MB
Name                    1825
Variety                11618
Origin                 11642
Attr_1                  5118
Attr_2                354153
Alcohol_vol             1892
Alcohol_percentage      1892
Winemaker_notes       334232
Review                336911
Avg_rating            386112
N_ratings             386112
Price_Out-of-stock      1893
Price                 341433
dtype: int64


# A1
Based on the provided `df.info()` and `df.isnull().sum()` outputs, the top three most critical data quality issues you need to address are the high number of missing values in key columns, the lack of data for critical price and rating metrics, and inconsistent data types for price information.

***

### 1. High Number of Missing Values in Key Columns 📉

The most significant issue is the sheer volume of **missing data** across several important columns, particularly `Avg_rating`, `N_ratings`, `Price`, `Winemaker_notes`, `Review`, and `Attr_2`. For example, over **89%** of the `Avg_rating` and `N_ratings` rows are missing (386,112 out of 432,659). 

**Why it's critical:** Your analysis is likely to focus on wine attributes, ratings, and pricing. With so much data missing, any analysis on these columns will be severely biased and incomplete. You can't draw reliable conclusions about a wine's quality or value if you're missing rating and price information for nearly **90%** of your dataset. This makes it impossible to perform meaningful statistical analysis, such as correlating price with ratings or identifying top-rated wines. Simply dropping these rows would reduce your dataset to only about 46,000 entries, which is a massive loss of information.

***

### 2. Inconsistent Data Types for Price Information 💰

The `Price` and `Price_Out-of-stock` columns are listed as `object` (string) data types, while they should be numerical (`float` or `int`). You can't perform mathematical calculations, such as calculating average prices or comparing prices, on columns that are stored as strings. This indicates a problem with how the data was collected or ingested, likely due to the presence of currency symbols (e.g., "$", "€") or other non-numeric characters within the data.

**Why it's critical:** You cannot analyze or model anything related to price until this is fixed. The `object` type prevents any direct quantitative analysis on these columns. You must first clean these columns by removing non-numeric characters and then convert them to a numerical format to enable any form of price-based analysis, such as price distribution analysis, regression modeling, or even simple statistical summaries.

***

### 3. Lack of Data for Core Analysis Metrics 🍷

The columns `Avg_rating`, `N_ratings`, and `Price` are arguably the most important for any commercial or quality-focused wine analysis. However, as noted above, a significant majority of your records are missing these values. The `Winemaker_notes` and `Review` columns, while also heavily missing, provide rich text data that could be used for sentiment analysis or content-based recommendations, but their absence for most of the dataset severely limits their utility.

**Why it's critical:** These columns represent the **core metrics** for evaluating wine quality and value. Without them, your analysis is confined to descriptive information like `Name`, `Variety`, and `Origin`. You can't answer critical business questions like: "What is the average price of a Cabernet Sauvignon?", "Do wines with more reviews have higher ratings?", or "Which regions produce the best-rated wines?". The data is primarily descriptive, lacking the quantitative metrics needed for deeper, more actionable insights.



# Q2
Based on your analysis, I need to fix the 'Price' and 'Price_Out-of-stock' columns with 'object' type. These columns contain strings or text.

Please provide a robust Python function:
1. Input: single string value.
2. Uses regular expressions to find and extract the  number from the string.
3. Return: the number as a float. If no valid number found, return a NaN.

# A2
A robust Python function to clean and convert your 'Price' and 'Price\_Out-of-stock' columns is provided below. This function uses regular expressions to handle a variety of string formats and will return a null value (`NaN`) if no number can be found.

```python
import re
import numpy as np

def extract_price(price_string):
    """
    Extracts a valid numeric price (integer or float) from a string.

    Args:
        price_string (str): The string to parse.

    Returns:
        float: The extracted number, or NaN if no valid number is found.
    """
    if not isinstance(price_string, str):
        return np.nan

    # Regular expression to find the first number (integer or float)
    # This pattern looks for a sequence of digits, with an optional decimal part.
    match = re.search(r'[\d.]+', price_string)

    if match:
        try:
            # Get the matched string and convert it to a float
            return float(match.group(0))
        except ValueError:
            # Return NaN if conversion to float fails
            return np.nan
    else:
        # Return NaN if no number is found
        return np.nan
```

### How it works:

1.  **Input Check**: The function first checks if the input is a string. If not, it returns `np.nan` to handle non-string data types gracefully.
2.  **Regular Expression**: It uses `re.search(r'[\d.]+', price_string)` to find the first sequence of digits (`\d`) and decimal points (`.`) in the string. This is a very robust pattern that can correctly identify numbers like `'99.99'` and `'150'` while ignoring surrounding text or symbols.
3.  **Extraction and Conversion**: If a number is found, it extracts the matched portion of the string using `match.group(0)` and attempts to convert it to a `float`.
4.  **Error Handling**: The `try...except` block handles any `ValueError` that might occur if the matched string (e.g., just `'.'`) cannot be converted to a float, returning `np.nan` instead.
5.  **Null Value Return**: If the regular expression finds no numbers at all, the function returns `np.nan`, which is the standard way to represent a null or missing numerical value in Python with `numpy` and `pandas`.

You can apply this function to your DataFrame columns using the `.apply()` method, as shown in the example below:

```python
# Assuming your DataFrame is named df
# df['Price'] = df['Price'].apply(extract_price)
# df['Price_Out-of-stock'] = df['Price_Out-of-stock'].apply(extract_price)
```

# Q3
I have a pandas DataFrame called correlation_matrix that stores the correlation values between my numerical features.
Please provide the Python code to create a heatmap from this matrix. The heatmap should have the correlation values displayed on each cell.

# A3
To create a heatmap of your correlation matrix with the values displayed on each cell, you can use seaborn's `heatmap()` function in combination with matplotlib.

### Python Code for Heatmap

Here is the complete Python code to generate the heatmap. This code assumes you have a pandas DataFrame named `correlation_matrix` and that `seaborn` and `matplotlib.pyplot` are imported.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create the heatmap
plt.figure(figsize=(10, 8)) # Optional: Adjust the size of the figure
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Add a title
plt.title('Correlation Matrix Heatmap')

# Display the plot
plt.show()
```

-----

### Code Explanation 📊

  * `sns.heatmap()`: This is the core function from the seaborn library.
      * `correlation_matrix`: This is the data you're plotting. The function takes your DataFrame directly.
      * `annot=True`: This crucial parameter **displays the correlation value** on each cell of the heatmap.
      * `cmap='coolwarm'`: This sets the color map. 'coolwarm' is a good choice for correlations because it uses a diverging color scheme, with a neutral color for zero correlation, and distinct colors for positive and negative correlations. Other options include 'viridis', 'plasma', or 'YlGnBu'.
      * `fmt=".2f"`: This formats the numbers displayed on the heatmap to two decimal places, making the plot cleaner and easier to read.
  * `plt.figure(figsize=(10, 8))`: This line creates a new figure and sets its size. It's optional but highly recommended for better readability, especially if you have many features.
  * `plt.title()`: This adds a clear title to your heatmap.
  * `plt.show()`: This displays the final plot.

This approach provides a clear and visually informative way to assess the relationships between your numerical features at a glance.




问：My 'Origin' column in pandas is inconsistent, with formats like 'Napa Valley, California, USA' and 'Bordeaux, France'. I need to extract the country to create a bar chart of the top 10 countries. What is a simple but effective strategy to do this in pandas for an initial analysis? Please provide the line of code to create a new 'Country' column based on this strategy.
答：
A simple and effective strategy to extract the country from the inconsistent 'Origin' column is to **split the string by the comma** and **take the last element**, as the country is consistently the final part of these location strings. You should then remove any leading or trailing whitespace from the resulting string.

```python
df['Country'] = df['Origin'].str.split(',').str[-1].str.strip()
```

This single line of code performs all the necessary steps:

  * `df['Origin'].str.split(',')`: This splits each string in the 'Origin' column by the comma, creating a list of substrings (e.g., `['Napa Valley', ' California', ' USA']`).
  * `.str[-1]`: This accesses the last element of each list, which is the country.
  * `.str.strip()`: This removes any extra whitespace, such as the space before " USA" in the example, ensuring clean country names.

After running this code, your `df['Country']` column will contain standardized country names like 'USA' and 'France', which you can then use to create your bar chart.

# Q4
I used a simple strategy to extract countries from an 'Origin' column by splitting by comma and take the last word. The strategy incorrectly identified 'California' as a country. But the U.S. states like 'California' should be mapped to 'United States'.
Acting as a data cleaning expert, what is a systematic approach to identify all possible U.S. states in my 'Country' column and correct them to 'United States'? I don't know the full list of states present in my data.

# A4
A systematic approach to identifying and correcting U.S. states is to **create a comprehensive list of all U.S. states and abbreviations** and use a mapping dictionary to perform the correction. This method is effective because it is both robust and scalable, ensuring you don't miss any states.

### Step 1: Obtain a List of U.S. States

First, you need a complete and accurate list of all U.S. states. You can either hard-code this list or, for a more robust solution, import it from a library or a reliable data source. This list should include both the full state names (e.g., 'California') and their two-letter postal abbreviations (e.g., 'CA') to cover various data entry formats.

### Step 2: Create a Mapping Dictionary

Once you have your list of states and abbreviations, you can create a dictionary where each state or abbreviation maps to 'United States'. This dictionary will be used to replace the incorrect country names.

### Step 3: Apply the Mapping to Your DataFrame

Use the pandas `.replace()` method on your 'Country' column, passing the dictionary you created in the previous step. This is a highly efficient way to replace multiple values simultaneously. Any value in the 'Country' column that is a key in your dictionary will be replaced with 'United States'.

-----

### Python Implementation

Here is a practical example of this systematic approach using Python and pandas.

```python
import pandas as pd

# Step 1: Create a comprehensive list of U.S. states and abbreviations.
# This list is a small sample for demonstration. You would need the full 50.
# A more robust solution would involve importing a complete list.
us_states = {
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
    'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
    'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts',
    'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska',
    'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina',
    'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
    'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming',
    # Include abbreviations as well for robustness
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN',
    'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE',
    'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
    'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
}

# Step 2: Create a mapping dictionary where states map to 'United States'.
state_mapping = {state: 'United States' for state in us_states}

# Step 3: Apply the mapping to your DataFrame.
# This assumes your DataFrame is named 'df'.
# The .replace() method will use the dictionary to find and replace values.
df['Country'] = df['Country'].replace(state_mapping)

print("Mapping complete. 'California' and other U.S. states are now 'United States'.")
```


# AI遗漏的问题：
# 发现国家里面有空字符串，删除
valid_country_counts = all_country_counts[all_country_counts.index != '']


# 发现！最后一列的Other U.S. 
print(top_10_valid_countries)
Country
United States    156018
France            88550
Italy             64589
Spain             25602
Australia         18636
Argentina         11401
South Africa      10084
Chile              9434
New Zealand        7790
Other U.S.         6468
Name: count, dtype: int64
# 查看'Country'为'Other U.S.'的行的原始'Origin'值
print(df[df['Country'] == 'Other U.S.']['Origin'].unique()[:10])
# 新增修正：
df['Country'] = df['Country'].replace('Other U.S.', 'United States')

# CYL使用的prompt

Begin by checking for any entries that are missing a winemaker note or have null values for the wine’s provenance.

Discard any records that do not contain a review.

Remove any non-numeric characters from the "Price-Out-of-Stock" and "Price" columns.

Convert the values in the "Price-Out-of-Stock" and "Price" columns to float types.

Combine the "Price-Out-of-Stock" and "Price" columns into a single feature named "Price_Feature."

Split any entries containing multiple locations in the "Origin" column into separate columns.

Label the separated columns from "Origin" as "Country" or "Region."

For wines from the United States that only list the state, assign "United States" as the country of origin.

Extract the year of production from the wine’s name if available.

Create a new feature named "Year" to store the extracted production year.

Ensure the final dataset contains exactly 2 text features, 7 categorical features, and 6 numerical features.

Create 6 histograms to display the distribution of each numerical feature in the cleaned dataset.

Redraw Figure 2 (Top ten countries) for the cleaned dataset and compare it to the raw dataset.

Redraw Figure 3 (Top ten grape varieties) for the cleaned dataset and compare it to the raw dataset.

Redraw Figure 5 (Average user rating of top ten countries) for the cleaned dataset and compare it to the raw dataset.

