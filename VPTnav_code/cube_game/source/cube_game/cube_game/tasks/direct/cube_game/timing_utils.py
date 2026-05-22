import numpy as np
from typing import Dict, List

def print_timing_report(timing_data: Dict) -> None:
    """
    Print comprehensive timing statistics from the pipeline.
    
    Args:
        timing_data: Dictionary containing timing information with structure:
            {
                'reset_idx_calls': List[float],
                'reset_internal_calls': List[float],
                'per_env': Dict[int, {
                    'setup_times': List[float],
                    'data_collection_attempts': List[float],
                    'total_time': float,
                    'start_time': float
                }]
            }
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE TIMING REPORT")
    print("="*80)
    
    # 1. Overall _reset_idx statistics
    if timing_data['reset_idx_calls']:
        reset_idx_times = timing_data['reset_idx_calls']
        print(f"\n📊 _reset_idx() Calls:")
        print(f"  Total calls: {len(reset_idx_times)}")
        print(f"  Total time: {sum(reset_idx_times):.2f}s")
        print(f"  Average per call: {np.mean(reset_idx_times):.2f}s")
        print(f"  Min: {min(reset_idx_times):.2f}s, Max: {max(reset_idx_times):.2f}s")
        print(f"  Std dev: {np.std(reset_idx_times):.2f}s")
    
    # 2. Overall _reset_idx_internal statistics
    if timing_data['reset_internal_calls']:
        reset_internal_times = timing_data['reset_internal_calls']
        print(f"\n📊 _reset_idx_internal() Calls:")
        print(f"  Total calls: {len(reset_internal_times)}")
        print(f"  Total time: {sum(reset_internal_times):.2f}s")
        print(f"  Average per call: {np.mean(reset_internal_times):.2f}s")
        print(f"  Min: {min(reset_internal_times):.2f}s, Max: {max(reset_internal_times):.2f}s")
        print(f"  Std dev: {np.std(reset_internal_times):.2f}s")
    
    # 3. Per-environment statistics
    if timing_data['per_env']:
        print(f"\n📊 Per-Environment Statistics:")
        print(f"  Total environments processed: {len(timing_data['per_env'])}")
        
        # Aggregate statistics
        all_total_times = []
        all_setup_times = []
        all_data_collection_times = []
        all_num_setup_attempts = []
        all_num_data_attempts = []
        
        for env_id, env_data in timing_data['per_env'].items():
            if env_data['total_time'] > 0:
                all_total_times.append(env_data['total_time'])
            if env_data['setup_times']:
                all_setup_times.extend(env_data['setup_times'])
                all_num_setup_attempts.append(len(env_data['setup_times']))
            if env_data['data_collection_attempts']:
                all_data_collection_times.extend(env_data['data_collection_attempts'])
                all_num_data_attempts.append(len(env_data['data_collection_attempts']))
        
        # Total time statistics (from creation to passing, excluding image collection)
        if all_total_times:
            print(f"\n  🕐 Total Time per Environment (creation → validation passed):")
            print(f"     Average: {np.mean(all_total_times):.2f}s")
            print(f"     Min: {min(all_total_times):.2f}s, Max: {max(all_total_times):.2f}s")
            print(f"     Median: {np.median(all_total_times):.2f}s")
            print(f"     Std dev: {np.std(all_total_times):.2f}s")
        
        # Setup time statistics
        if all_setup_times:
            print(f"\n  🔧 Setup Time per Attempt (spawn + validation, up to circle generation):")
            print(f"     Total attempts: {len(all_setup_times)}")
            print(f"     Average per attempt: {np.mean(all_setup_times):.2f}s")
            print(f"     Min: {min(all_setup_times):.2f}s, Max: {max(all_setup_times):.2f}s")
            print(f"     Median: {np.median(all_setup_times):.2f}s")
            print(f"     Std dev: {np.std(all_setup_times):.2f}s")
        
        if all_num_setup_attempts:
            print(f"\n  🔁 Number of Setup Attempts per Environment:")
            print(f"     Average: {np.mean(all_num_setup_attempts):.1f} attempts")
            print(f"     Min: {min(all_num_setup_attempts)}, Max: {max(all_num_setup_attempts)}")
            print(f"     Total setup attempts across all envs: {sum(all_num_setup_attempts)}")
        
        # Data collection statistics
        if all_data_collection_times:
            print(f"\n  📸 Data Collection Time per Attempt (circle validation):")
            print(f"     Total attempts: {len(all_data_collection_times)}")
            print(f"     Average per attempt: {np.mean(all_data_collection_times):.2f}s")
            print(f"     Min: {min(all_data_collection_times):.2f}s, Max: {max(all_data_collection_times):.2f}s")
            print(f"     Median: {np.median(all_data_collection_times):.2f}s")
            print(f"     Std dev: {np.std(all_data_collection_times):.2f}s")
        
        if all_num_data_attempts:
            print(f"\n  🔁 Number of Data Collection Attempts per Environment:")
            print(f"     Average: {np.mean(all_num_data_attempts):.1f} attempts")
            print(f"     Min: {min(all_num_data_attempts)}, Max: {max(all_num_data_attempts)}")
        
        # Detailed breakdown for slowest/fastest environments
        print(f"\n  🐌 Slowest 5 Environments (by total time):")
        sorted_envs = sorted(
            [(env_id, data['total_time']) for env_id, data in timing_data['per_env'].items() if data['total_time'] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for env_id, total_time in sorted_envs:
            env_data = timing_data['per_env'][env_id]
            num_setups = len(env_data['setup_times'])
            num_collections = len(env_data['data_collection_attempts'])
            avg_setup = np.mean(env_data['setup_times']) if env_data['setup_times'] else 0
            avg_collection = np.mean(env_data['data_collection_attempts']) if env_data['data_collection_attempts'] else 0
            print(f"     Env {env_id}: {total_time:.2f}s total | {num_setups} setup attempts (avg {avg_setup:.2f}s) | {num_collections} collection attempts (avg {avg_collection:.2f}s)")
        
        print(f"\n  ⚡ Fastest 5 Environments (by total time):")
        fastest_envs = sorted(
            [(env_id, data['total_time']) for env_id, data in timing_data['per_env'].items() if data['total_time'] > 0],
            key=lambda x: x[1]
        )[:5]
        for env_id, total_time in fastest_envs:
            env_data = timing_data['per_env'][env_id]
            num_setups = len(env_data['setup_times'])
            num_collections = len(env_data['data_collection_attempts'])
            avg_setup = np.mean(env_data['setup_times']) if env_data['setup_times'] else 0
            avg_collection = np.mean(env_data['data_collection_attempts']) if env_data['data_collection_attempts'] else 0
            print(f"     Env {env_id}: {total_time:.2f}s total | {num_setups} setup attempts (avg {avg_setup:.2f}s) | {num_collections} collection attempts (avg {avg_collection:.2f}s)")
    
    print("\n" + "="*80)


def save_timing_report_to_file(timing_data: Dict, filepath: str) -> None:
    """
    Save detailed timing report to a text file.
    
    Args:
        timing_data: Dictionary containing timing information
        filepath: Path to save the report
    """
    import sys
    from io import StringIO
    
    # Capture print output
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    print_timing_report(timing_data)
    
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    
    # Write to file
    with open(filepath, 'w') as f:
        f.write(output)
    
    print(f"Timing report saved to: {filepath}")


def export_timing_data_to_csv(timing_data: Dict, output_dir: str) -> None:
    """
    Export timing data to CSV files for further analysis.
    
    Args:
        timing_data: Dictionary containing timing information
        output_dir: Directory to save CSV files
    """
    import os
    import csv
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Export per-environment data
    env_csv_path = os.path.join(output_dir, 'per_env_timing.csv')
    with open(env_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'env_id', 'total_time', 'num_setup_attempts', 'avg_setup_time', 
            'num_data_collection_attempts', 'avg_data_collection_time'
        ])
        
        for env_id, data in timing_data['per_env'].items():
            num_setups = len(data['setup_times'])
            num_collections = len(data['data_collection_attempts'])
            avg_setup = np.mean(data['setup_times']) if data['setup_times'] else 0
            avg_collection = np.mean(data['data_collection_attempts']) if data['data_collection_attempts'] else 0
            
            writer.writerow([
                env_id, data['total_time'], num_setups, avg_setup, 
                num_collections, avg_collection
            ])
    
    print(f"Per-environment timing data saved to: {env_csv_path}")
    
    # Export reset_idx call times
    reset_idx_csv_path = os.path.join(output_dir, 'reset_idx_calls.csv')
    with open(reset_idx_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['call_number', 'duration_seconds'])
        for i, duration in enumerate(timing_data['reset_idx_calls']):
            writer.writerow([i, duration])
    
    print(f"reset_idx call times saved to: {reset_idx_csv_path}")
    
    # Export reset_internal call times
    reset_internal_csv_path = os.path.join(output_dir, 'reset_internal_calls.csv')
    with open(reset_internal_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['call_number', 'duration_seconds'])
        for i, duration in enumerate(timing_data['reset_internal_calls']):
            writer.writerow([i, duration])
    
    print(f"reset_internal call times saved to: {reset_internal_csv_path}")


# Example usage:
if __name__ == "__main__":
    # Example timing data structure
    example_timing_data = {
        'reset_idx_calls': [10.5, 12.3, 9.8],
        'reset_internal_calls': [2.1, 2.3, 1.9, 2.5],
        'per_env': {
            0: {
                'setup_times': [2.1, 1.8],
                'data_collection_attempts': [0.5, 0.6],
                'total_time': 5.0,
                'start_time': 0.0
            },
            1: {
                'setup_times': [2.3],
                'data_collection_attempts': [0.7],
                'total_time': 3.0,
                'start_time': 5.0
            }
        }
    }
    
    print_timing_report(example_timing_data)