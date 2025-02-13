import pandas as pd
import re
from dateutil import parser
import streamlit as st
import numpy as np
import itertools

def find_combinations(word_length_pairs, max_length):
    valid_combinations = []

    # Loop through combinations of all lengths (1 to total number of words)
    for r in range(1, 4):
        for combination in itertools.combinations(word_length_pairs, r):
            # Calculate the total length of this combination
            total_length = sum(pair[1] for pair in combination)

            # If it satisfies the length condition, add the words to the result
            if (max_length - 0) <= total_length <= (max_length + 0):
                valid_combinations.append([pair[0] for pair in combination])

    return valid_combinations
def find_invoice_indices(text, keywords):
    """
    Find the indices of words matching specific patterns in the text.

    Parameters:
        text (str): Input text to analyze.

    Returns:
        list: List of indices where the keywords are found.
    """
    # Define the keywords to search for

    # Split the text into words
    words = text.split()

    # Find indices of words that contain any of the keywords (case-insensitive)
    matching_indices = [
        index for index, word in enumerate(words)
        if any(keyword.lower() in word.lower() for keyword in keywords)
    ]

    return matching_indices
def get_surrounding_indices(lst, indices, window=20):
    """
    Extracts surrounding numbers within a given window size for each index in a list.

    Args:
        lst (list): The original list of numbers or elements.
        indices (list): Indices around which to extract numbers.
        window (int): Number of elements to extract before and after each index.

    Returns:
        dict: A dictionary with the target index as the key and the surrounding numbers as the value.
    """
    result = {}
    for index in indices:
        start = max(0, index - window)  # Ensure we don't go out of bounds
        end = min(len(lst), index + window + 1)  # Ensure we don't exceed the list size
        result[index] = lst[start:end]
    return result
def find_irn(text, keywords):
    text = text.lower()
    irn_indices = find_invoice_indices(text, keywords)
    surrounding_numbers = get_surrounding_indices(text.split(), irn_indices, window=20)
    # print(surrounding_numbers)
    values = []
    irn_number = []
    matches = None
    found = False
    for key, value in surrounding_numbers.items():
        for val in value:
            values.append(val)
            matches = re.findall(r'[a-f0-9]{64,}', val)
            if len(matches) > 0:
                irn_number = matches  # Get the first match
                found = True  # Set the flag
                break  # Break inner loop
        if found:  # Check the flag and break outer loop
            break
    sorted_values_a_f_0_9, sorted_values_a_f_0_9_length = [], []
    if len(irn_number) == 0:
        sorted_values = sorted(values, key=len, reverse=True)
        for v in sorted_values:
            pattern = re.fullmatch(r'[0-9a-f]+-?', v, re.IGNORECASE)
            if pattern and len(v)>=6:
                v = v.replace("-", "")
                sorted_values_a_f_0_9.append(v)
                sorted_values_a_f_0_9_length.append(len(v))
            else:
                pass
        # print(sorted_values_a_f_0_9)
        # print(sorted_values_a_f_0_9_length)

        max_length = 64 # Maximum length of the IRN
        # Create a list of (word, length) pairs
        word_length_pairs = list(zip(sorted_values_a_f_0_9, sorted_values_a_f_0_9_length))

        # Find valid combinations
        combinations = find_combinations(word_length_pairs, max_length)
        # Print the results
        # print(f"Total valid combinations: {len(combinations)}")
        if len(combinations) > 0:
            for combo in combinations:
                combo_index =[]
                for co in combo:
                    combo_index.append([x.replace("-", "") for x in text.split()].index(co))
                sorted_combo = [x for _, x in sorted(zip(combo_index, combo))]
                irn_number = [''.join(sorted_combo)]
    if len(irn_number) > 0:
        e_invoice = True
    else:
        e_invoice = False
    if len(irn_number) > 0:
        df = pd.DataFrame({"key": ["IRN_No", "E_Invoice"], "value": [irn_number[0], e_invoice]})
    else:
        df = pd.DataFrame({"key": ["IRN_No", "E_Invoice"], "value": ["Not there in document", e_invoice]})
    return df
def format_indian_currency(amount):
    amount = float(amount)
    if isinstance(amount, (int, float)):
        return f"₹{amount:,.0f}"
    return amount
def format_percentage(value):
    try:
        if isinstance(value, str) and value.endswith('%'):
            # Remove '%' and convert to float
            numeric_value = float(value.strip('%'))
            # Format as percentage with two decimal places
            return f"{numeric_value:.3f}%"
        elif isinstance(value, str):
            # Remove '%' and convert to float
            numeric_value = float(value)
            # Format as percentage with two decimal places
            return f"{numeric_value:.3f}%"
        elif isinstance(value, (int, float)):
            # Directly format numeric values
            return f"{value:.3f}%"
    except ValueError:
        return value  # Return the original value if conversion fails
    return value
def apply_transformations(df):
    """
    Applies transformations to the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to apply transformations on.

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    # Work on a copy of the DataFrame
    df = df.copy()

    commands = [
        "df.loc[df['key'] == 'Gross_Amount', 'value'] = df.loc[df['key'] == 'Gross_Amount', 'value'].apply(format_indian_currency)",
        "df.loc[df['key'] == 'Basic_Amount', 'value'] = df.loc[df['key'] == 'Basic_Amount', 'value'].apply(format_indian_currency)",
        "df.loc[df['key'] == 'Tax_Amount', 'value'] = df.loc[df['key'] == 'Tax_Amount', 'value'].apply(format_indian_currency)",
        "df.loc[df['key'] == 'CGST_Amount', 'value'] = df.loc[df['key'] == 'CGST_Amount', 'value'].apply(format_indian_currency)",
        "df.loc[df['key'] == 'SGST_Amount', 'value'] = df.loc[df['key'] == 'SGST_Amount', 'value'].apply(format_indian_currency)",
        "df.loc[df['key'] == 'IGST_Amount', 'value'] = df.loc[df['key'] == 'IGST_Amount', 'value'].apply(format_indian_currency)",
        "df.loc[df['key'] == 'CGST_Tax_Rate', 'value'] = df.loc[df['key'] == 'CGST_Tax_Rate', 'value'].apply(format_percentage)",
        "df.loc[df['key'] == 'SGST_Tax_Rate', 'value'] = df.loc[df['key'] == 'SGST_Tax_Rate', 'value'].apply(format_percentage)",
        "df.loc[df['key'] == 'IGST_Tax_Rate', 'value'] = df.loc[df['key'] == 'IGST_Tax_Rate', 'value'].apply(format_percentage)"
    ]

    for command in commands:
        try:
            exec(command)
        except Exception as e:
            print(f"Error executing command: {command}")
            print(f"Error: {e}")

    return df

# Apply transformations
# transformed_df = apply_transformations(finaldf)

def convert_to_number(amount_str):
    amount_str = str(amount_str)
    cleaned_str = re.sub(r'[^\d.]', '', amount_str)  # Remove non-numeric and non-decimal characters
    cleaned_str = re.sub(r'\.(?=.*\.)', '', cleaned_str)  # Remove all dots except the first one

    # Remove commas
    cleaned_str = cleaned_str.replace(',', '')

    # Check if it's a valid number after cleaning
    if re.match(r'^\d+(\.\d+)?$', cleaned_str):
        # Convert to float and return as int if no decimal part exists
        numeric_value = float(cleaned_str)
        return int(numeric_value) if numeric_value.is_integer() else numeric_value
    else:
        return amount_str
def convert_to_ddmmyyyy(input_date):
    try:
        # Parse the input date
        date_obj = parser.parse(input_date, dayfirst=True)  # Ensures day-first interpretation
        # Format the date as "DD.MM.YYYY"
        return date_obj.strftime("%d.%m.%Y")
    except:
        return convert_to_not_mentioned_insource(input_date)
def keep_only_numerics(input_str):
    if input_str is None:
        return "Not Mentioned in Source Document"
    else:
        return re.sub(r'[^0-9]', '', input_str)
def convert_to_zero(str):
    if str is None:
        return "0"
    str_converted = str.lower()
    str_converted = str_converted.strip()
    if str_converted.startswith("none"):
        return "0"
    elif str_converted.find("none") != -1:
        return "0"
    elif str_converted.find("not") != -1:
        return "0"
    elif str_converted=="":
        return "0"
    elif str_converted.find("not mention") != -1:
        return "0"
    elif str_converted.find("n/a") != -1:
        return "0"
    else:
        return str
def convert_to_not_mentioned_insource(string):
    if string is None:
        return "Not Mentioned in Source Document"
    if isinstance(string, bool):
        return str(string)
    str_converted = string.lower()
    str_converted = str_converted.strip()
    if str_converted.find("none") != -1:
        return "Not Mentioned in Source Document"
    elif str_converted.find("not mention") != -1:
        return "Not Mentioned in Source Document"
    elif str_converted.find("not") != -1:
        return "Not Mentioned in Source Document"
    elif str_converted=="":
        return "Not Mentioned in Source Document"
    elif str_converted.find("n/a") != -1:
        return "Not Mentioned in Source Document"
    else:
        return string
def extract_currency_type(str):
    if str is None:
        return "Not Mentioned in Source Document"
    str_converted = str.lower()
    str_converted = str_converted.strip()
    if str_converted.startswith("inr"):
        return "INR"
    elif str_converted.find("rupe") != -1:
        return "INR"
    elif str_converted.find("rs") != -1:
        return "INR"
    elif str_converted.find("inr") != -1:
        return "INR"
    elif str_converted.find("india") != -1:
        return "INR"
    elif str_converted.find("₹") != -1:
        return "INR"
    elif str_converted.find("not mention") != -1:
        return "Not Mentioned in Source Document"
    elif str_converted.find("none") != -1:
        return "Not Mentioned in Source Document"
    elif str_converted.find("n/a") != -1:
        return "Not Mentioned in Source Document"
    elif str_converted=="":
        return "Not Mentioned in Source Document"
    else:
        return str
def convert_to_decimal(input_str):
    # Convert to float if it's a valid number
    try:
        input_str = re.sub(r'[^.0-9]', '', input_str)
        input_str = float(input_str)
        if input_str>=1:
            return input_str/100
        return float(input_str)
    except ValueError:
        input_str = 0
        return input_str
def keypairs_df_conversion(keypairs_df):
    # keypairs_df.loc[keypairs_df["key"] == "HSN_SAC_No", "value"] = keypairs_df.loc[keypairs_df["key"] == "HSN_SAC_No", "value"].apply(keep_only_numerics)
    keypairs_df.loc[keypairs_df["key"] == "Currency", "value"] = keypairs_df.loc[keypairs_df["key"] == "Currency", "value"].apply(extract_currency_type)
    keypairs_df.loc[keypairs_df["key"] == "Gross_Amount", "value"] = keypairs_df.loc[keypairs_df["key"] == "Gross_Amount", "value"].apply(convert_to_zero)
    keypairs_df.loc[keypairs_df["key"] == "Basic_Amount", "value"] = keypairs_df.loc[keypairs_df["key"] == "Basic_Amount", "value"].apply(convert_to_zero)
    keypairs_df.loc[keypairs_df["key"] == "Tax_Amount", "value"] = keypairs_df.loc[keypairs_df["key"] == "Tax_Amount", "value"].apply(convert_to_zero)
    keypairs_df.loc[keypairs_df["key"] == "CGST_Tax_Rate", "value"] = keypairs_df.loc[keypairs_df["key"] == "CGST_Tax_Rate", "value"].apply(convert_to_zero)
    keypairs_df.loc[keypairs_df["key"] == "SGST_Tax_Rate", "value"] = keypairs_df.loc[keypairs_df["key"] == "SGST_Tax_Rate", "value"].apply(convert_to_zero)
    keypairs_df.loc[keypairs_df["key"] == "IGST_Tax_Rate", "value"] = keypairs_df.loc[keypairs_df["key"] == "IGST_Tax_Rate", "value"].apply(convert_to_zero)
    for col in ["CGST_Amount", "SGST_Amount", "IGST_Amount"]:
        if col in keypairs_df["key"].values:
            keypairs_df.loc[keypairs_df["key"] == col, "value"] = keypairs_df.loc[keypairs_df["key"] == col, "value"].apply(convert_to_zero)
    keypairs_df.loc[keypairs_df["key"] == "Gross_Amount", "value"] = keypairs_df.loc[keypairs_df["key"] == "Gross_Amount", "value"].apply(convert_to_number)
    keypairs_df.loc[keypairs_df["key"] == "Basic_Amount", "value"] = keypairs_df.loc[keypairs_df["key"] == "Basic_Amount", "value"].apply(convert_to_number)
    keypairs_df.loc[keypairs_df["key"] == "Tax_Amount", "value"] = keypairs_df.loc[keypairs_df["key"] == "Tax_Amount", "value"].apply(convert_to_number)
    for col in ["CGST_Amount", "SGST_Amount", "IGST_Amount"]:
        if col in keypairs_df["key"].values:
            keypairs_df.loc[keypairs_df["key"] == col, "value"] = keypairs_df.loc[keypairs_df["key"] == col, "value"].apply(convert_to_number)
    keypairs_df.loc[keypairs_df["key"] == "Invoice_Date", "value"] = keypairs_df.loc[keypairs_df["key"] == "Invoice_Date", "value"].apply(convert_to_ddmmyyyy)
    #########################
    if is_number(keypairs_df.loc[keypairs_df["key"] == "CGST_Amount", "value"]) or is_number(keypairs_df.loc[keypairs_df["key"] == "SGST_Amount", "value"]):
        keypairs_df.loc[keypairs_df["key"] == "IGST_Amount", "value"] = "0"
        keypairs_df.loc[keypairs_df["key"] == "IGST_Tax_Rate", "value"] = "0"
    elif is_number(keypairs_df.loc[keypairs_df["key"] == "IGST_Amount", "value"]):
        keypairs_df.loc[keypairs_df["key"] == "CGST_Amount", "value"] = "0"
        keypairs_df.loc[keypairs_df["key"] == "SGST_Amount", "value"] = "0"
        keypairs_df.loc[keypairs_df["key"] == "CGST_Tax_Rate", "value"] = "0"
        keypairs_df.loc[keypairs_df["key"] == "SGST_Tax_Rate", "value"] = "0"
    ##############################
    keypairs_df.loc[keypairs_df["key"] == "Vendor_Supplier_GSTN", "value"] = keypairs_df.loc[keypairs_df["key"] == "Vendor_Supplier_GSTN", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "Buyer_GSTN", "value"] = keypairs_df.loc[keypairs_df["key"] == "Buyer_GSTN", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "E_Invoice", "value"] = keypairs_df.loc[keypairs_df["key"] == "E_Invoice", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "HSN_SAC_No", "value"] = keypairs_df.loc[keypairs_df["key"] == "HSN_SAC_No", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "IRN_No", "value"] = keypairs_df.loc[keypairs_df["key"] == "IRN_No", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "RCM_Applicability", "value"] = keypairs_df.loc[keypairs_df["key"] == "RCM_Applicability", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "Nature Of Goods & Services", "value"] = keypairs_df.loc[keypairs_df["key"] == "Nature Of Goods & Services", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "PO_SO_Number", "value"] = keypairs_df.loc[keypairs_df["key"] == "PO_SO_Number", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "Supplier_PAN", "value"] = keypairs_df.loc[keypairs_df["key"] == "Supplier_PAN", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "Supplier_Name", "value"] = keypairs_df.loc[keypairs_df["key"] == "Supplier_Name", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "Buyer_Receiver_Name", "value"] = keypairs_df.loc[keypairs_df["key"] == "Buyer_Receiver_Name", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "TCS Amount", "value"] = keypairs_df.loc[keypairs_df["key"] == "TCS Amount", "value"].apply(convert_to_not_mentioned_insource)
    keypairs_df.loc[keypairs_df["key"] == "Is_Company", "value"] = keypairs_df.loc[keypairs_df["key"] == "Is_Company", "value"].apply(convert_to_not_mentioned_insource)
    ###################
    keypairs_df = keypairs_df.astype(str)
    return keypairs_df
def linetable_validation(linetable_df):
    for col in ["IGST Rate", "CGST Rate", "SGST Rate", "CGST Amount", "SGST Amount", "IGST Amount", "Total Amount", "Price", "Quantity"]:
        if col in linetable_df.columns:
            # Convert the column to string type to avoid errors when using .str methods
            linetable_df[col] = linetable_df[col].astype(str)

            # Replace 'None' or other invalid values with '0'
            linetable_df[col] = linetable_df[col].replace({None: "0", "None": "0", "none": "0", "": "0", "n/a": "0"})

            # Use .str accessor to handle other string patterns like 'not mention'
            linetable_df.loc[linetable_df[col].str.lower().str.contains("not mention", na=False), col] = "0"
            linetable_df.loc[linetable_df[col].str.lower().str.contains("none", na=False), col] = "0"
            linetable_df.loc[linetable_df[col].str.lower().str.contains("n/a", na=False), col] = "0"
    for col in ["CGST Amount", "SGST Amount", "IGST Amount", "Total Amount", "Price"]:
        if col in linetable_df.columns:
            linetable_df[col] = linetable_df[col].apply(convert_to_number)
            linetable_df[col] = linetable_df[col].apply(format_indian_currency)
    for col in ["CGST Rate", "SGST Rate", "IGST Rate"]:
        if col in linetable_df.columns:
            linetable_df[col] = linetable_df[col].apply(format_percentage)
    for col in ["HSN/SAC Code", "Item_Description", "Item_Code"]:
        if col in linetable_df.columns:
            # Convert the column to string type to avoid errors when using .str methods
            linetable_df[col] = linetable_df[col].astype(str)

            # Replace 'None' or other invalid values with 'Not Mentioned in Source Document'
            linetable_df[col] = linetable_df[col].replace({None: "Not Mentioned in Source Document", "None": "Not Mentioned in Source Document", "none": "Not Mentioned in Source Document", "": "Not Mentioned in Source Document", "n/a": "Not Mentioned in Source Document"})

            # Use .str accessor to handle other string patterns like 'not mention'
            linetable_df.loc[linetable_df[col].str.lower().str.contains("not mention", na=False), col] = "Not Mentioned in Source Document"
            linetable_df.loc[linetable_df[col].str.lower().str.contains("none", na=False), col] = "Not Mentioned in Source Document"
            linetable_df.loc[linetable_df[col].str.lower().str.contains("n/a", na=False), col] = "Not Mentioned in Source Document"

    # total_sum = linetable_df.select_dtypes(include=['number']).sum(axis=0)
    # total_amount1 = 0
    # for col in ["CGST Amount", "SGST Amount", "IGST Amount", "Price"]:
    #     if col in linetable_df.columns:
    #         total_amount1 = total_amount1 + total_sum[col]
    # total_amount2 = total_sum["Total Amount"]


    # if abs(total_amount1/total_amount2-1)>0.05:
    #     linetable_df["Quantity"] = linetable_df["Quantity"].apply(convert_to_number)
    #     linetable_df["Net Amount"] = linetable_df["Price"] * linetable_df["Quantity"]
    # else:
    #     linetable_df["Net Amount"] = linetable_df["Price"]


    # for col in ["CGST Rate", "SGST Rate", "IGST Rate"]:
    #     if col in linetable_df.columns:
    #         linetable_df[col+"_new"] = linetable_df[col].apply(convert_to_decimal)
    #         linetable_df[col+" Amount_new"] = linetable_df[col+"_new"] * linetable_df["Net Amount"]
    # linetable_df["Total Amount_new"] = linetable_df["Net Amount"]
    # for col in ["CGST Rate", "SGST Rate", "IGST Rate"]:
    #     if col in linetable_df.columns:
    #         linetable_df["Total Amount_new"] = linetable_df["Total Amount_new"] + linetable_df[col+" Amount_new"]

    # total_sum = linetable_df.select_dtypes(include=['number']).sum(axis=0)

    # if "CGST Rate" in linetable_df.columns:
    #     if ((sum(linetable_df["CGST Rate Amount_new"]-linetable_df["CGST Amount"]) <100) and
    #         (sum(linetable_df["SGST Rate Amount_new"] - linetable_df["SGST Amount"]) < 100) and
    #             (sum(linetable_df["Total Amount_new"] - linetable_df["Total Amount_new"]) < 100)):
    #         print("Tax amounts fetched properly")
    #     else:
    #         print("Tax amounts not fetched properly")
    # if "IGST Rate" in linetable_df.columns:
    #     if ((sum(linetable_df["IGST Rate Amount_new"] - linetable_df["IGST Amount"]) < 100) and
    #             (sum(linetable_df["Total Amount_new"] - linetable_df["Total Amount_new"]) < 100)):
    #         print("Tax amounts fetched properly")
    #     else:
    #         print("Tax amounts not fetched properly")

    # linetable_df = pd.concat([linetable_df, pd.concat([linetable_df.select_dtypes(include=['number']).sum(axis=0).rename('Total_Sum')]).to_frame().T])
    #
    # Tax= 0
    # for col in ["CGST Amount", "SGST Amount", "IGST Amount"]:
    #     if col in linetable_df.columns:
    #         print(f'{col}: {total_sum[col]}')
    #         Tax = Tax + total_sum[col]
    # print(f'Net Amount: {total_sum["Net Amount"]}')
    # print(f'Total Amount: {total_sum["Total Amount"]}')
    # print(f'Total Tax: {Tax}')
    # print(f'Total Tax Rate: {Tax / total_sum["Net Amount"] * 100}%')
    linetable_df = linetable_df.astype(str)
    return linetable_df
def is_number(value):
    try:
        float(value)  # Attempt to convert to a float
        if float(value) != 0:
            return True
        else:
            return False
    except ValueError:
        return False
def check_amounts(keypairs_df):
    df = keypairs_df.copy()
    txt = ""
    df = df.loc[df['key'].isin(
        ['Gross_Amount', 'Basic_Amount', 'Tax_Amount', 'CGST_Amount', 'SGST_Amount', 'IGST_Amount','CGST_Tax_Rate', 'SGST_Tax_Rate','IGST_Tax_Rate']), ['key', 'value']]
    df.loc[df["key"] == "CGST_Tax_Rate", "value"] = df.loc[df["key"] == "CGST_Tax_Rate", "value"].apply(convert_to_decimal)
    df.loc[df["key"] == "SGST_Tax_Rate", "value"] = df.loc[df["key"] == "SGST_Tax_Rate", "value"].apply(convert_to_decimal)
    df.loc[df["key"] == "IGST_Tax_Rate", "value"] = df.loc[df["key"] == "IGST_Tax_Rate", "value"].apply(convert_to_decimal)
    df['value'] = df['value'].astype(float)
    Gross_Amount = df.loc[df['key'].isin(['Gross_Amount']), 'value'].tolist()[0]
    Basic_Amount = df.loc[df['key'].isin(['Basic_Amount']), 'value'].tolist()[0]
    Tax_Amount = df.loc[df['key'].isin(['Tax_Amount']), 'value'].tolist()[0]
    CGST_Amount = df.loc[df['key'].isin(['CGST_Amount']), 'value'].tolist()[0]
    SGST_Amount = df.loc[df['key'].isin(['SGST_Amount']), 'value'].tolist()[0]
    IGST_Amount = df.loc[df['key'].isin(['IGST_Amount']), 'value'].tolist()[0]
    CGST_Tax_Rate = df.loc[df['key'].isin(['CGST_Tax_Rate']), 'value'].tolist()[0]
    SGST_Tax_Rate = df.loc[df['key'].isin(['SGST_Tax_Rate']), 'value'].tolist()[0]
    IGST_Tax_Rate = df.loc[df['key'].isin(['IGST_Tax_Rate']), 'value'].tolist()[0]

    df.loc[df["key"] == "CGST_Tax_Rate", "value"] = f"{df.loc[df['key'] == 'CGST_Tax_Rate', 'value'].tolist()[0] * 100}%"
    df.loc[df["key"] == "SGST_Tax_Rate", "value"] = f"{df.loc[df['key'] == 'SGST_Tax_Rate', 'value'].tolist()[0] * 100}%"
    df.loc[df["key"] == "IGST_Tax_Rate", "value"] = f"{df.loc[df['key'] == 'IGST_Tax_Rate', 'value'].tolist()[0] * 100}%"

    if (Gross_Amount - Basic_Amount) > 0:
        if abs(Gross_Amount/(Basic_Amount+Tax_Amount)-1)<0.001:
            txt = txt + "Correct: Gross_Amount = Basic_Amount + Tax_Amount\n"
        else:
            Tax_Amount = Gross_Amount - Basic_Amount
            df.loc[df["key"] == "Tax_Amount", "value"] = Tax_Amount
        if abs(Tax_Amount - (CGST_Amount+SGST_Amount + IGST_Amount)) < 5:
            txt = txt + "Correct: Tax_Amount = CGST_Amount + SGST_Amount + IGST_Amount\n"
            if CGST_Amount>0:
                if abs(CGST_Amount/(Basic_Amount * CGST_Tax_Rate)-1)<0.001:
                    txt = txt + "Correct: CGST_Amount = Basic_Amount * CGST_Tax_Rate\n"
                else:
                    txt = txt + "InCorrect: CGST_Amount <> Basic_Amount * CGST_Tax_Rate\n"
            else:
                txt = txt + "Correct: CGST_Amount = 0\n"
            if SGST_Amount > 0:
                if abs(SGST_Amount/(Basic_Amount * SGST_Tax_Rate)-1)<0.001:
                    txt = txt + "Correct: SGST_Amount = Basic_Amount * SGST_Tax_Rate\n"
                else:
                    txt = txt + "InCorrect: SGST_Amount <> Basic_Amount * SGST_Tax_Rate\n"
            else:
                txt = txt + "Correct: SGST_Amount = 0\n"
            if IGST_Amount > 0:
                if abs(IGST_Amount/(Basic_Amount * IGST_Tax_Rate)-1)<0.001:
                    txt = txt + "Correct: IGST_Amount = Basic_Amount * IGST_Tax_Rate\n"
                else:
                    txt = txt + "InCorrect: IGST_Amount <> Basic_Amount * IGST_Tax_Rate\n"
            else:
                txt = txt + "Correct: IGST_Amount = 0\n"
        elif (Tax_Amount - (CGST_Amount + SGST_Amount + IGST_Amount)) > 5:
            txt = txt + "InCorrect: Some issues in Tax Amounts! Tax_Amount <> CGST_Amount + SGST_Amount + IGST_Amount\n"
    else:
        txt = txt + "InCorrect: Some issues in Gross Amounts! Gross_Amount <> Basic_Amount + Tax_Amount\n"
    return df, txt
def convert_amountdf(amountdf):
    # amountdf.reset_index(inplace=True)
    amountdf = amountdf[['key', 'value']]

    Gross_Amount = amountdf.loc[amountdf['key'].isin(['Gross_Amount']), 'value'].tolist()[0]
    Basic_Amount = amountdf.loc[amountdf['key'].isin(['Basic_Amount']), 'value'].tolist()[0]
    Tax_Amount = amountdf.loc[amountdf['key'].isin(['Tax_Amount']), 'value'].tolist()[0]
    CGST_Amount = amountdf.loc[amountdf['key'].isin(['CGST_Amount']), 'value'].tolist()[0]
    SGST_Amount = amountdf.loc[amountdf['key'].isin(['SGST_Amount']), 'value'].tolist()[0]
    IGST_Amount = amountdf.loc[amountdf['key'].isin(['IGST_Amount']), 'value'].tolist()[0]
    CGST_Tax_Rate = amountdf.loc[amountdf['key'].isin(['CGST_Tax_Rate']), 'value'].tolist()[0]
    SGST_Tax_Rate = amountdf.loc[amountdf['key'].isin(['SGST_Tax_Rate']), 'value'].tolist()[0]
    IGST_Tax_Rate = amountdf.loc[amountdf['key'].isin(['IGST_Tax_Rate']), 'value'].tolist()[0]
    CGST_Tax_Rate = convert_to_decimal(CGST_Tax_Rate)
    SGST_Tax_Rate = convert_to_decimal(SGST_Tax_Rate)
    IGST_Tax_Rate = convert_to_decimal(IGST_Tax_Rate)

    # Gross_Amount = 0
    # Basic_Amount = 1000
    # Tax_Amount = 250
    # CGST_Amount, SGST_Amount, IGST_Amount = 0, 0, 0
    # CGST_Tax_Rate, SGST_Tax_Rate, IGST_Tax_Rate = 0, 0, 0
    if Tax_Amount==0 :
        if abs(Tax_Amount - (CGST_Amount + SGST_Amount + IGST_Amount))==0:
            if CGST_Tax_Rate>0 or SGST_Tax_Rate>0 or IGST_Tax_Rate>0:
                CGST_Amount = Basic_Amount * CGST_Tax_Rate
                SGST_Amount = Basic_Amount * SGST_Tax_Rate
                IGST_Amount = Basic_Amount * IGST_Tax_Rate
                Tax_Amount = CGST_Amount + SGST_Amount + IGST_Amount
            else:
                if Gross_Amount==0:
                    Gross_Amount = Basic_Amount + Tax_Amount
        elif abs(Tax_Amount - (CGST_Amount + SGST_Amount + IGST_Amount)) > 0:
            Tax_Amount = CGST_Amount + SGST_Amount + IGST_Amount
            CGST_Tax_Rate = CGST_Amount / Basic_Amount
            SGST_Tax_Rate = SGST_Amount / Basic_Amount
            IGST_Tax_Rate = IGST_Amount / Basic_Amount
        Gross_Amount = Basic_Amount + Tax_Amount
    else:
        if (Tax_Amount - (CGST_Amount + SGST_Amount + IGST_Amount)) > 1:
            if CGST_Tax_Rate>0:
                CGST_Amount = Basic_Amount * CGST_Tax_Rate
            if SGST_Tax_Rate>0:
                SGST_Amount = Basic_Amount * SGST_Tax_Rate
            if IGST_Tax_Rate>0:
                IGST_Amount = Basic_Amount * IGST_Tax_Rate
            if CGST_Amount==0 and SGST_Amount==0 and IGST_Amount==0:
                IGST_Amount = Tax_Amount
                IGST_Tax_Rate = IGST_Amount / Basic_Amount
            Tax_Amount = CGST_Amount + SGST_Amount + IGST_Amount
            CGST_Tax_Rate = CGST_Amount / Basic_Amount
            SGST_Tax_Rate = SGST_Amount / Basic_Amount
            IGST_Tax_Rate = IGST_Amount / Basic_Amount
            if abs((Tax_Amount - (CGST_Amount + SGST_Amount + IGST_Amount)))<0:
                print("good")
            else:
                print("bad")
        elif abs(Tax_Amount - (CGST_Amount + SGST_Amount + IGST_Amount)) < 1:
            CGST_Tax_Rate = CGST_Amount / Basic_Amount
            SGST_Tax_Rate = SGST_Amount / Basic_Amount
            IGST_Tax_Rate = IGST_Amount / Basic_Amount
        else:
            Tax_Amount = CGST_Amount + SGST_Amount + IGST_Amount
            CGST_Tax_Rate = CGST_Amount / Basic_Amount
            SGST_Tax_Rate = SGST_Amount / Basic_Amount
            IGST_Tax_Rate = IGST_Amount / Basic_Amount
            Gross_Amount = Basic_Amount + Tax_Amount
        Gross_Amount = Basic_Amount + Tax_Amount

    df = pd.DataFrame({'key': ['Gross_Amount', 'Basic_Amount', 'Tax_Amount', 'CGST_Amount', 'SGST_Amount', 'IGST_Amount', 'CGST_Tax_Rate', 'SGST_Tax_Rate', 'IGST_Tax_Rate'],
                         'value': [Gross_Amount, Basic_Amount, Tax_Amount, CGST_Amount, SGST_Amount, IGST_Amount, CGST_Tax_Rate*100, SGST_Tax_Rate*100, IGST_Tax_Rate*100]})
    df = df.astype(str)
    return df


def replace_amount_rows(transformed_df, transformed_convert_amountdf):
    key_columns = [
		"Gross_Amount", "Basic_Amount", "Tax_Amount",
		"CGST_Amount", "SGST_Amount", "IGST_Amount",
		"CGST_Tax_Rate", "SGST_Tax_Rate", "IGST_Tax_Rate"
	]
    # Remove rows from transformed_df where the 'key' column matches any in the list
    transformed_df = transformed_df[~transformed_df['key'].isin(key_columns)]

    # Get rows from transformed_convert_amountdf where the 'key' column matches
    rows_to_add = transformed_convert_amountdf[transformed_convert_amountdf['key'].isin(key_columns)]

    # Append the filtered rows to transformed_df
    transformed_df = pd.concat([transformed_df, rows_to_add], ignore_index=True)

    # Verify results
    return transformed_df

def replace_irn(transformed_df, transformed_convert_irndf):
    key_columns = ["IRN_No", "E_Invoice"]
    # Remove rows from transformed_df where the 'key' column matches any in the list
    transformed_df = transformed_df[~transformed_df['key'].isin(key_columns)]

    # Get rows from transformed_convert_amountdf where the 'key' column matches
    rows_to_add = transformed_convert_irndf[transformed_convert_irndf['key'].isin(key_columns)]

    # Append the filtered rows to transformed_df
    transformed_df = pd.concat([transformed_df, rows_to_add], ignore_index=True)

    # Verify results
    return transformed_df


def rcm_applicability(transformed_convert_amountdf):
    try:
        Tax_Amount = transformed_convert_amountdf.loc[transformed_convert_amountdf["key"] == "Tax_Amount", "value"].tolist()[0]
        Tax_Amount = convert_to_zero(Tax_Amount)
        Tax_Amount = convert_to_number(Tax_Amount)
        if Tax_Amount>0:
            transformed_convert_amountdf.loc[transformed_convert_amountdf["key"] == "RCM_Applicability", "value"] = "No"
        else:
            transformed_convert_amountdf.loc[transformed_convert_amountdf["key"] == "RCM_Applicability", "value"] = "Yes"
    except:
        transformed_convert_amountdf.loc[transformed_convert_amountdf["key"] == "RCM_Applicability", "value"] = "Unknown"
    return transformed_convert_amountdf


