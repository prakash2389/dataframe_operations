import pandas as pd


def rename_variables(df, column_name):
    """
    Rename values in a specified column by appending _1, _2, etc., if duplicates exist.
    This function ensures each value is unique by tracking occurrences and updating accordingly.

    Parameters:
    df (pd.DataFrame): A DataFrame containing the column to rename.
    column_name (str): The name of the column to process.

    Returns:
    pd.DataFrame: DataFrame with a new column '<column_name>_renamed' containing unique values.
    """
    value_counts = {}
    new_values = []

    for value in df[column_name]:
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

        new_value = f"{value}_{value_counts[value]}"
        new_values.append(new_value)

    df[f'{column_name}_renamed'] = new_values
    return df


# Sample data
data = {
    'variable': [
        'item_id', 'ITEM_DESCRIPTION', 'HSN_SAC_CODE', 'QUANTITY', 'UNIT_PRICE', 'LINE_ITEM_PRICE', 'ITEM_DESCRIPTION'
    ],
    'value': [2, 'C226I COPIER/PRINTER', 997314, 1.000, 4400.00, 4400.00, 'adsdsadsa']
}

df = pd.DataFrame(data)

# Apply function
df = rename_variables(df, 'variable')

# Display result
print(df)

#            variable                 value    variable_renamed
# 0           item_id                     2           item_id_1
# 1  ITEM_DESCRIPTION  C226I COPIER/PRINTER  ITEM_DESCRIPTION_1
# 2      HSN_SAC_CODE                997314      HSN_SAC_CODE_1
# 3          QUANTITY                   1.0          QUANTITY_1
# 4        UNIT_PRICE                4400.0        UNIT_PRICE_1
# 5   LINE_ITEM_PRICE                4400.0   LINE_ITEM_PRICE_1
# 6  ITEM_DESCRIPTION             adsdsadsa  ITEM_DESCRIPTION_2
