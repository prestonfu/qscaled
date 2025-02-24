import matplotlib.pyplot as plt
import seaborn as sns


def plot_per_batch_size(df, n_utds, n_envs, batch_sizes, thresholds):
    for batch_size in batch_sizes:
        colors = sns.color_palette("viridis", n_colors=len(df['learning_rate'].unique()))
        fig, axs = plt.subplots(n_utds, n_envs, figsize=(3.5*n_envs, 2.5*n_utds))
        fig.suptitle(f'Learning Curves by Environment and UTD Ratio (Batch Size = {batch_size})')

        lines = []
        labels = []
        config_colors = {}
        color_idx = 0

        # Filter data for current batch size
        df_filtered = df[df['batch_size'] == batch_size]

        # Group data by environment and UTD ratio
        for i, env in enumerate(sorted(df['env_name'].unique())):
            env_data = df_filtered[df_filtered['env_name'] == env]
            
            # Create separate plots for each UTD value
            for j, utd in enumerate(sorted(env_data['utd'].unique())):
                utd_data = env_data[env_data['utd'] == utd]
                # Sort by learning rate only since we're already filtering by UTD
                utd_data = utd_data.sort_values('learning_rate')
                
                ax = axs[j, i]
                ax.set_title(f'{env} (UTD={utd})')
                
                for _, row in utd_data.iterrows():
                    config = row['learning_rate']
                    if config not in config_colors:
                        config_colors[config] = color_idx
                        color_idx += 1
                    
                    label = f"lr={row['learning_rate']}"
                    line = ax.plot(row['training_step'], row['mean_return'], alpha=0.3, color=colors[config_colors[config]])
                    line = ax.plot(row['training_step'], row['return_isotonic'], alpha=1, color=colors[config_colors[config]])
                    
                    # use the crossings column to plot crossings
                    for k, threshold in enumerate(thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        ax.plot(crossing_x, crossing_y, 'o', color=colors[config_colors[config] % len(colors)])
                    
                    # Plot crossing standard deviations as error bars
                    for k, threshold in enumerate(thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        crossing_std = row['crossings_std'][k]
                        ax.errorbar(crossing_x, crossing_y, xerr=crossing_std, fmt='none', color=colors[config_colors[config] % len(colors)], capsize=3)

                    # Only add to legend if we haven't seen this combination before
                    if label not in labels:
                        lines.append(line[0])
                        labels.append(label)
                
                ax.set_xlabel('Steps')
                ax.set_ylabel('Return')
                ax.grid(True)
                ax.set_facecolor('#f0f0f0')

        # Sort labels by learning rate
        sorted_indices = [i for i, _ in sorted(enumerate(labels), 
                                            key=lambda x: float(x[1].replace("lr=","")))]
        lines = [lines[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]

        # Create a single legend outside all subplots
        fig.legend(lines, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=(len(labels)), fontsize=12)
        plt.tight_layout()
        plt.show()
 

def plot_per_lr(df, n_utds, n_envs, learning_rates, thresholds):
    for lr in sorted(learning_rates):
        colors = sns.color_palette("viridis", n_colors=len(df['batch_size'].unique()))  # Using viridis for a nice gradient
        fig, axs = plt.subplots(n_utds, n_envs, figsize=(3.5*n_envs, 2.5*n_utds))
        fig.suptitle(f'Learning Curves by Environment and UTD Ratio (Learning Rate = {lr})')

        lines = []
        labels = []
        config_colors = {}
        color_idx = 0

        # Filter data for current learning rate
        df_filtered = df[df['learning_rate'] == lr]

        # Group data by environment and UTD ratio
        for i, env in enumerate(sorted(df['env_name'].unique())):
            env_data = df_filtered[df_filtered['env_name'] == env]
            
            # Create separate plots for each UTD value
            for j, utd in enumerate(sorted(df['utd'].unique())):
                utd_data = env_data[env_data['utd'] == utd]
                utd_data = utd_data.sort_values('batch_size')
                
                ax = axs[j, i]
                ax.set_title(f'{env} (UTD={utd})')
                
                for _, row in utd_data.iterrows():
                    config = row['batch_size']
                    if config not in config_colors:
                        config_colors[config] = color_idx
                        color_idx += 1
                    
                    label = f"batch_size={row['batch_size']}"
                    line = ax.plot(row['training_step'], row['mean_return'], alpha=0.3, color=colors[config_colors[config] % len(colors)])
                    line = ax.plot(row['training_step'], row['return_isotonic'], alpha=1, color=colors[config_colors[config] % len(colors)])

                    # use the crossings column to plot crossings
                    for k, threshold in enumerate(thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        ax.plot(crossing_x, crossing_y, 'o', color=colors[config_colors[config] % len(colors)])
                        
                    # Plot crossing standard deviations as error bars
                    for k, threshold in enumerate(thresholds):
                        crossing_x = row['crossings'][k]
                        crossing_y = threshold
                        crossing_std = row['crossings_std'][k]
                        ax.errorbar(crossing_x, crossing_y, xerr=crossing_std, fmt='none', color=colors[config_colors[config] % len(colors)], capsize=3)
                    
                    # Only add to legend if we haven't seen this combination before
                    if label not in labels:
                        lines.append(line[0])
                        labels.append(label)
                
                ax.set_xlabel('Steps')
                ax.set_ylabel('Return')
                ax.grid(True)
                ax.set_facecolor('#f0f0f0')

        # Sort labels by batch size
        sorted_indices = [i for i, _ in sorted(enumerate(labels), 
                                            key=lambda x: float(x[1].replace("batch_size=","")))]
        lines = [lines[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices] 

        # Create a single legend outside all subplots
        fig.legend(lines, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=(len(labels)), fontsize=12)
        plt.tight_layout()
        plt.show()
