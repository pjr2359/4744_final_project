"""
aggregate_runs.py
Collate accuracy across models, few‑shot sizes, and K runs.
"""

import glob, re, json, pathlib
try:
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: Required packages missing. Please install with:")
    print("pip install pandas matplotlib")
    exit(1)

def flatten(path):
    m, n, r = re.search(r"_(.+?)_n(\d+)_run(\d+)", path).groups()
    rows=[]
    data=json.load(open(path))
    for _id, items in data.items():
        for obj in items:
            # Convert llm_label to a proper boolean if it's a string or handle "ERROR"
            llm_label = obj["llm_label"]
            if isinstance(llm_label, str):
                if llm_label == "ERROR":
                    continue  
                llm_label = llm_label.lower().startswith('t')
            
            rows.append(dict(model=m, n=int(n), run=int(r),
                             id=_id, test=obj["test_string"],
                             llm=llm_label))
    return rows

def main():
    files=glob.glob("cache/llm_labels_*_n*_run*.json")
    if not files:
        print("no files")
        return
    
    # Group files by model for debugging
    model_files = {}
    for f in files:
        match = re.search(r"_(.+?)_n(\d+)_run(\d+)", f)
        if match:
            model = match.group(1)
            model_files.setdefault(model, []).append(f)
    
    print("Files per model:")
    for model, files_list in model_files.items():
        print(f"  {model}: {len(files_list)} files")
    
    
    files = [f for f in files if "flash-preview" not in f]
    print(f"\nAnalyzing {len(files)} files (excluding flash-preview)")
    
    df=pd.DataFrame([x for f in files for x in flatten(f)])
    
    # Count samples per model-n-run combination to filter out incomplete runs
    sample_counts = df.groupby(['model', 'n', 'run']).size().reset_index(name='count')
    print("\nSample counts per model-n-run:")
    print(sample_counts)
    
    # Set a minimum threshold for valid runs (adjust as needed)
    MIN_SAMPLES = 100
    valid_runs = sample_counts[sample_counts['count'] >= MIN_SAMPLES]
    print(f"\nKeeping only runs with at least {MIN_SAMPLES} samples:")
    print(valid_runs[['model', 'n', 'run', 'count']])
    
    # Filter dataframe to only include valid runs
    df = df.merge(valid_runs[['model', 'n', 'run']], on=['model', 'n', 'run'])
    print(f"\nFiltered data shape: {df.shape}")
    
   
    print("\nData shape:", df.shape)
    print("\nModel distribution:")
    print(df['model'].value_counts())
    
    # Check for strange values
    if 'llm' in df.columns:
        print("\nLLM label types:")
        print(df['llm'].apply(type).value_counts())
        
        # Print a sample of values for each model
        print("\nSample values by model:")
        for model in df['model'].unique():
            sample = df[df['model'] == model].sample(min(5, df[df['model'] == model].shape[0]))
            print(f"\n{model} sample:")
            print(sample[['n', 'run', 'llm']])

    # bring in author labels once
    gold={}
    with open("data/pooled_truth.txt") as fh:
        for ln in fh:
            parts=ln.strip().split()
            g=[]
            for s,l in zip(parts[1::2], parts[2::2]):
                if l.lower() in ("t","f"):
                    g.append((s, l.lower()=="t"))
            gold[parts[0]]=dict(g)

    df["author"]=df.apply(lambda row: gold[row.id][row.test], axis=1)
    df["correct"]=df["llm"]==df["author"]

    summary=(df.groupby(["model","n","run"])["correct"].mean()
               .reset_index(name="accuracy"))
    
    #  debugging prints
    print("\nAccuracy per model, n, run:")
    for model in summary['model'].unique():
        print(f"\n{model}:")
        print(summary[summary['model'] == model][['n', 'run', 'accuracy']])
    
    csv="fewshot_accuracy_runs.csv"
    summary.to_csv(csv,index=False)
    print("wrote",csv)

    # average over runs
    avg=(summary.groupby(["model","n"])["accuracy"].mean()
         .unstack())  # n as columns
    
    # Print the raw averaged data
    print("\nAverage accuracy across runs (raw data):")
    print(avg)
    
    avg.to_csv("fewshot_accuracy_mean.csv")

    # Only use N values that are common to all models
    common_n_values = set.intersection(*[set(avg.loc[model].dropna().index) for model in avg.index])
    print(f"Using common N values across all models: {sorted(common_n_values)}")
    
    # Create a complete dataset for connected lines
    filtered_avg = avg[sorted(common_n_values)]
    
    # Get all available n values for each model for individual points
    model_n_values = {model: set(avg.loc[model].dropna().index) for model in avg.index}
    all_n_values = sorted(set.union(*model_n_values.values()))
    print(f"All N values found: {all_n_values}")

    plt.figure(figsize=(10,5))
    
    # Completely rewrite the plotting logic with explicit x and y values
    for model in sorted(avg.index):
       
        if "flash-preview" in model:
            continue
        
        # Get valid pairs of (n, accuracy) for this model
        x_values = []
        y_values = []
        for n in sorted(model_n_values[model]):
            if not pd.isna(avg.loc[model, n]):
                x_values.append(n)
                y_values.append(avg.loc[model, n])
        
        # Print the data points for this model
        print(f"\nPlotting {model} points: x={x_values}, y={y_values}")
        
        # Map the model names to more readable labels
        model_label = model
        if model == "labels_flash":
            model_label = "2.0 flash"
        elif model == "labels_flash-lite":
            model_label = "2.0 flash lite"
        elif model == "labels_pro":
            model_label = "1.5 pro"
        
        # Use different styles for different models
        if 'preview' in model:
            # For flash-preview, use green color with diamond markers
            if len(x_values) >= 2:
                # With multiple points, connect them with lines
                plt.plot(x_values, y_values, '-D', color='green', linewidth=2, 
                         markersize=8, label=model)
            else:
                # With single point, just show the marker
                plt.plot(x_values, y_values, 'D', color='green', markersize=10,
                         label=f"{model} (n={x_values[0]})")
        else:
            # For other models, use default styling with circles
            plt.plot(x_values, y_values, '-o', linewidth=2, label=model_label)
    
    # Set y-axis to reasonable range based on data
    plt.ylim(0.6, 0.72)  # Smaller range to highlight differences
    
    # Force x axis to use integer ticks at the actual n-values
    plt.xticks(sorted(all_n_values))
    
    plt.xlabel("# few‑shot examples")
    plt.ylabel("Accuracy (avg of K runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fewshot_curve.png",dpi=150)
    print("wrote fewshot_curve.png")

if __name__=="__main__":
    main()
