def summary_to_markdown_list(summary_text):
    """
    Parses the output of model.summary() and converts it to a Markdown list.
    """
    lines = summary_text.splitlines()
    md_lines = ["# Model Architecture Summary", ""]  # Start with a header

    # Find the start and end of the table. We skip the header and footer lines.
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if line.startswith('_'):
            if start_idx is None:
                start_idx = i + 1  # Line after the top rule is the header
            else:
                end_idx = i  # Bottom rule is the end of the table
                break
        # Also catch the "Total params" line which is after the table
        if line.startswith("Total params"):
            end_idx = i
            break

    # Extract the table body and the totals
    if start_idx and end_idx:
        # The header is at start_idx, the data rows are from start_idx+1 to end_idx-1
        # Let's just get all lines from the body, skipping the header rule
        table_body = lines[start_idx+1:end_idx]
        
        # Process each layer line
        for line in table_body:
            # Skip empty lines and the final line of the table (which is often a rule)
            if line.strip() and not line.startswith('='):
                # Split the line by spaces, then filter out empty strings
                parts = [p for p in line.split(' ') if p != '']
                # The structure is: [Layer Name, (Layer Type), (Output Shape), Param #]
                # We need to re-join the layer name and type in case they have spaces
                # A better approach is to split based on the fixed-width nature
                try:
                    # This is a simpler method: split by the known column dividers
                    # Name/Type is roughly first 25 chars, Output Shape next 20, Params last
                    name_type = line[:25].strip()
                    output_shape = line[25:45].strip()
                    params = line[45:].strip()
                    
                    # Further split the first part to get name and type
                    if '(' in name_type:
                        name = name_type.split(' (')[0]
                        layer_type = name_type.split(' (')[1].rstrip(')')
                    else:
                        name = name_type
                        layer_type = "Unknown"
                    
                    md_lines.append(f"*   **{name}** ({layer_type})")
                    md_lines.append(f"    *   **Output Shape:** `{output_shape}`")
                    md_lines.append(f"    *   **Parameters:** {params}")
                except IndexError:
                    # If parsing fails, just add the raw line as a bullet point
                    md_lines.append(f"*   {line.strip()}")
    
    # Now add the Total params lines which are after the table
    for line in lines[end_idx:]:
        if line.strip():  # If it's not an empty line
            # Format lines like "Total params: 101,770" as bold
            if ':' in line:
                key, value = line.split(':', 1)
                md_lines.append(f"**{key.strip()}:** {value.strip()}")
            else:
                md_lines.append(f"**{line}**")
    
    return "\n".join(md_lines)
