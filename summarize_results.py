import os
import yaml
import pandas as pd

def extract_last_row(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        if not lines:
            return None
        return lines[-1].split()

def extract_yaml_fields(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def collect_experiment_data(root_dir):
    all_rows = []

    for subdir, dirs, files in os.walk(root_dir):
        if {'opt.yaml', 'results.txt', 'test_results.txt'}.issubset(files):
            try:
                opt = extract_yaml_fields(os.path.join(subdir, 'opt.yaml'))
                results = extract_last_row(os.path.join(subdir, 'results.txt'))
                test_results = extract_last_row(os.path.join(subdir, 'test_results.txt'))

                row = dict(opt)  # start with opt.yaml fields

                if results and len(results) >= 4:
                    row.update({
                        'epochs': results[0],
                        'auc_val': results[1],
                        'train_losses': results[2],
                        'valid_losses': results[3],
                    })

                if test_results and len(test_results) >= 3:
                    row.update({
                        'test_epoch': test_results[0],
                        'test_accuracy': test_results[1],
                        'auc_test': test_results[2],
                    })

                row['experiment'] = os.path.basename(subdir)  # optional: folder name
                all_rows.append(row)

            except Exception as e:
                print(f"Error in {subdir}: {e}")

    return pd.DataFrame(all_rows)

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else './runs'
    df = collect_experiment_data(root)
    df.to_csv(os.path.join(root, 'summary.csv'), index=False)
    print("Summary saved to summary.csv")
