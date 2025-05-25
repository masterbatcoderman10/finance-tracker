# CORRECTED VERSION - No infinite loop
# This version fixes the potential infinite loop issues in the original code

def process_table_safely(table_re):
    """
    Safely process table data without risk of infinite loops
    """
    processed_rows = []
    ix = 1  # Start from 1 to skip header if needed
    max_ix = len(table_re)  # Use full length, not len(table_re[1:])

    while ix < max_ix:
        row = table_re[ix]
        if any(row):
            date = None
            description = None
            debits = None
            credits = None
            balance = None

            row_found = False

            # CRITICAL FIX: Add bounds check to inner loop
            while not row_found and ix < max_ix:
                potential_date = row[0]
                potential_description = row[1]
                potential_debits = row[2]
                potential_credits = row[3]
                potential_balance = row[4]

                if potential_date and (potential_debits or potential_credits or potential_balance):
                    # This is complete row
                    processed_rows.append({
                        'date': potential_date,
                        'description': potential_description,
                        'debits': potential_debits,
                        'credits': potential_credits,
                        'balance': potential_balance
                    })
                    ix += 1
                    row_found = True
                elif potential_date and potential_description:
                    # This is the start of a new row
                    date = potential_date
                    description = potential_description
                    ix += 1
                    # CRITICAL FIX: Check bounds before accessing array
                    if ix < max_ix:
                        row = table_re[ix]
                    else:
                        # Handle incomplete row at end of data
                        processed_rows.append({
                            'date': date,
                            'description': description,
                            'debits': None,
                            'credits': None,
                            'balance': None
                        })
                        row_found = True
                elif potential_description and (potential_debits or potential_credits or potential_balance):
                    # This is the end of a row
                    description = (description or '') + \
                        ' ' + potential_description
                    processed_rows.append({
                        'date': date,
                        'description': description,
                        'debits': potential_debits,
                        'credits': potential_credits,
                        'balance': potential_balance
                    })
                    ix += 1
                    row_found = True
                elif (potential_debits or potential_credits or potential_balance):
                    # This is an alternate end of a row
                    processed_rows.append({
                        'date': date,
                        'description': description,
                        'debits': potential_debits,
                        'credits': potential_credits,
                        'balance': potential_balance
                    })
                    ix += 1
                    row_found = True
                else:
                    # This is the end of a row or unrecognized pattern
                    print("reached odd exception")
                    print(row)
                    ix += 1
                    row_found = True
        else:
            # IMPORTANT: Handle empty rows by incrementing ix
            ix += 1

    return processed_rows

# Key fixes made:
# 1. Added `and ix < max_ix` to the inner while loop condition
# 2. Added bounds checking before accessing `table_re[ix]`
# 3. Fixed max_ix calculation to use len(table_re) instead of len(table_re[1:])
# 4. Added handling for empty rows in the else clause
# 5. Added proper handling for incomplete rows at the end of data
# 6. Fixed potential None concatenation issue with `(description or '')`


print("Corrected code loaded. Use process_table_safely(table_re) to process your data safely.")
