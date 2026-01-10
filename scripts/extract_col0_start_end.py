import os
import csv
import glob

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def main():
    output_dir = 'output'
    summary_file = 'col0_start_end.csv'
    
    # Get all csv files
    files = glob.glob(os.path.join(output_dir, '*.csv'))
    files.sort() # Ensure deterministic order
    
    print(f"Found {len(files)} CSV files in {output_dir}")
    
    with open(summary_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['start', 'end'])
        
        count = 0
        for file_path in files:
            # Skip the summary file if it ends up in the same directory
            if os.path.basename(file_path) == summary_file:
                continue
                
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Filter for data lines (skip comments and headers)
                data_lines = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split(',')
                    # Check if the first element is a number (time) to identify data lines
                    # The header starts with 'time', which is not a float.
                    if len(parts) > 1 and is_number(parts[0]):
                        data_lines.append(parts)
                
                if data_lines:
                    first_row = data_lines[0]
                    last_row = data_lines[-1]
                    
                    # col0 is at index 1 based on header: time,col0,col1...
                    # Ensure we have enough columns
                    if len(first_row) > 1 and len(last_row) > 1:
                        start_val = first_row[1]
                        end_val = last_row[1]
                        
                        writer.writerow([start_val, end_val])
                        count += 1
                    else:
                        print(f"Skipping {file_path}: Not enough columns")
                else:
                    # print(f"Skipping {file_path}: No data lines found")
                    pass
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
    print(f"Successfully processed {count} files. Results written to {summary_file}")

if __name__ == "__main__":
    main()
