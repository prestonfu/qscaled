import os
import numpy as np
import pandas as pd

from qscaled.core.preprocessing import get_envs, get_utds, get_batch_sizes, get_learning_rates


def tabulate_proposed_params(
    utds_to_predict, proposed_lr_values, proposed_bs_values, outputs_dir, save_path, verbose=False
):
    # TODO: MAKE THIS CLEANER
    envs = proposed_lr_values['Environment']

    # Display merged table of proposed values
    if verbose:
        print('\nProposed Values:')
        print('-' * 160)
        print(f"{'Environment':<30} {'UTD':<10} {'Learning Rate':<15} {'Learning Rate x√2':<15} {'Learning Rate x√0.5':<15} {'Batch Size':<15} {'Batch Size x√2':<15} {'Batch Size x√0.5':<15}")
        print('-' * 160)

    proposed_values_formatted = []

    for env in sorted(np.unique(envs)):
        for utd in utds_to_predict:
            utd = f'{utd:.2f}'
            # Find indices for this env/UTD combination
            satisfying_bs_idx = (
                i for i in range(len(proposed_bs_values['Environment']))
                if proposed_bs_values['Environment'][i] == env and proposed_bs_values['UTD'][i] == utd
            )
            bs_idx = next(satisfying_bs_idx, None)
            
            satisfying_lr_idx = (
                i for i in range(len(proposed_lr_values['Environment']))
                if proposed_lr_values['Environment'][i] == env and proposed_lr_values['UTD'][i] == utd
            )
            lr_idx = next(satisfying_lr_idx, None)

            if lr_idx is not None and bs_idx is not None:
                if verbose:
                    print(
                        f"{env:<30} {utd:<10} "
                        f"{proposed_lr_values['Learning Rate'][lr_idx]:<15} "
                        f"{proposed_lr_values['Learning Rate x√2'][lr_idx]:<15} "
                        f"{proposed_lr_values['Learning Rate x√0.5'][lr_idx]:<15} "
                        f"{proposed_bs_values['Batch Size'][bs_idx]:<15} "
                        f"{proposed_bs_values['Batch Size x√2'][bs_idx]:<15} "
                        f"{proposed_bs_values['Batch Size x√0.5'][bs_idx]:<15}"
                    )

                proposed_values_formatted.append(
                    {
                        'Environment': env,
                        'UTD': utd,
                        'Learning Rate': proposed_lr_values['Learning Rate'][lr_idx],
                        'Learning Rate x√2': proposed_lr_values['Learning Rate x√2'][lr_idx],
                        'Learning Rate x√0.5': proposed_lr_values['Learning Rate x√0.5'][lr_idx],
                        'Batch Size': proposed_bs_values['Batch Size'][bs_idx],
                        'Batch Size x√2': proposed_bs_values['Batch Size x√2'][bs_idx],
                        'Batch Size x√0.5': proposed_bs_values['Batch Size x√0.5'][bs_idx],
                    }
                )

    proposed_values_df = pd.DataFrame(proposed_values_formatted).astype(
        {
            'Environment': str,
            'UTD': float,
            'Learning Rate': float,
            'Learning Rate x√2': float,
            'Learning Rate x√0.5': float,
            'Batch Size': int,
            'Batch Size x√2': int,
            'Batch Size x√0.5': int,
        }
    )

    for c in proposed_values_df.columns:
        if 'Batch Size' in c:
            proposed_values_df[f'{c}(rounded)'] = (np.round(proposed_values_df[c] / 16) * 16).astype(int)

    outfile = os.path.join(outputs_dir, 'grid_proposed_hparams', f'{save_path}_fitted.csv')
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    proposed_values_df.to_csv(outfile, index=False)
    return proposed_values_df


def tabulate_baseline_params(utds_to_predict, utd, df, outputs_dir, save_path):
    """
    For a fixed UTD (geometric mean among predicted UTDs), find the best
    (batch size, learning rate) pair. Then run this across many UTDs. The
    output csv details additional experiments to be run.
    
    If utd == 'middle', the UTD is set to the geometric mean of the grid search.
    """
    utds = get_utds(df)
    envs = get_envs(df)
    n_envs = len(envs)
    if utd == 'middle':
        middle_utd = np.prod(utds_to_predict) ** (1/len(utds_to_predict))  # Geometric mean of predicted UTDs
        utd = min(utds, key=lambda x: abs(np.log(x) - np.log(middle_utd)))  # Snap to nearest UTD in grid search
    print('Baseline based on UTD', utd)
    
    utd_data = df[df['utd'] == utd]
    utd_data['last_crossing'] = utd_data['crossings'].apply(lambda x: x[-1])
    idx = utd_data.groupby(['env_name'])['last_crossing'].idxmin()
    baseline_values_df = utd_data.loc[idx]

    res_dict = {}
    for _, row in baseline_values_df.iterrows():
        res_dict[row['env_name']] = {
            'batch_size': row['batch_size'],
            'learning_rate': row['learning_rate'],
        }

    baseline_values_df['utd'] = [utds_to_predict] * n_envs
    baseline_values_df = baseline_values_df.explode('utd').reset_index(drop=True)
    baseline_values_df = baseline_values_df[['env_name', 'utd', 'learning_rate', 'batch_size']].rename(
        columns={
            'env_name': 'Environment',
            'utd': 'UTD',
            'learning_rate': 'Learning Rate',
            'batch_size': 'Batch Size',
        }
    )

    hparam_dir = os.path.join(outputs_dir, 'grid_proposed_hparams')
    os.makedirs(hparam_dir, exist_ok=True)
    base_fname = f'{save_path}_baseline_utd{utd}'
    
    # Save configurations for existing and extrapolated UTDs separately
    baseline_values_df.query(f'UTD in {utds}').to_csv(
        f'{hparam_dir}/{base_fname}_existing.csv', index=False
    )
    baseline_values_df.query(f'UTD not in {utds}').to_csv(
        f'{hparam_dir}/{base_fname}_new.csv', index=False
    )

    return baseline_values_df
