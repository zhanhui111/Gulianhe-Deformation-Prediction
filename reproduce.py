"""Portable verification/inference and optional retraining of the main model.

python -B reproduce.py                 # all three 2021 lead protocols
python -B reproduce.py --train         # retrain locked main model, then verify
No downloads. New run outputs go only to generated/ inside this package.
"""
from pathlib import Path
import os, sys, json, pickle, argparse, platform, time
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import torch
from core import load_points, metrics
from multihorizon import Thermal, make_pair, dataset
from anchored_bp import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    started = time.perf_counter()
    out = ROOT/'generated'
    out.mkdir(exist_ok=True)
    lock = json.loads((ROOT/'results/anchored_bp/locked_selection.json').read_text())
    if args.train:
        d, dates, X, S, meta = dataset()
        tr = np.flatnonzero(meta.role.eq('train'))
        if len(tr) > 240000:
            tr = np.sort(np.random.default_rng(42).choice(tr,240000,replace=False))
        model = fit(X.loc[tr], (meta.y-meta['last']).to_numpy()[tr],
                    hidden=lock['hidden'], steps=lock['steps'], members=5, ga=True)
        with (out/'retrained_main_model.pkl').open('wb') as f:
            pickle.dump(model,f,protocol=5)
        del X, S, meta
    else:
        with (ROOT/'results/anchored_bp/Anchored GA-PSO-BP.pkl').open('rb') as f:
            model = pickle.load(f)
        d, dates = load_points()
    expected = pd.read_csv(ROOT/'results/anchored_bp/metrics.csv')
    expected = expected[expected.model.eq('Shrink Anchored GA-PSO-BP')].set_index('horizon_min_days')
    thermal = Thermal(d)
    scores, checks, feature_names = [], [], None
    for horizon in [30,60,90]:
        obs, predictions, leads = [], [], []
        for target_col in dates:
            target = pd.Timestamp(target_col[2:])
            eligible = [c for c in dates if pd.Timestamp(c[2:]) <= target-pd.Timedelta(days=horizon)]
            if target.year != 2021 or len(eligible) < 4:
                continue
            origin = pd.Timestamp(eligible[-1][2:])
            if origin < pd.Timestamp('2020-11-01'):
                continue
            X, _, meta = make_pair(d,dates,origin.strftime('%Y-%m-%d'),target.strftime('%Y-%m-%d'),thermal)
            feature_names = list(X.columns)
            pred = meta['last'].to_numpy()+lock['shrink_alpha']*model.predict(X)
            obs.append(meta.y.to_numpy())
            predictions.append(pred)
            leads.append(int((target-origin).days))
        row = dict(horizon_min_days=horizon, actual_lead_min=min(leads),
                   actual_lead_max=max(leads), **metrics(np.concatenate(obs),np.concatenate(predictions)))
        scores.append(row)
        error = max(abs(row[k]-float(expected.loc[horizon,k])) for k in ['R2','RMSE','MAE','bias'])
        passed = bool(error < 1e-4 and row['n'] == int(expected.loc[horizon,'n']))
        checks.append(dict(horizon_min_days=horizon, matches_archived_metrics=passed,
                           maximum_metric_difference=error))
        print(f'{horizon} d protocol ({min(leads)}--{max(leads)} d): '
              f'R2={row["R2"]:.6f}, RMSE={row["RMSE"]:.6f} mm; archived match={passed}',flush=True)
    pd.DataFrame(scores).to_csv(out/'reproduced_metrics.csv',index=False,float_format='%.10g')
    report = dict(action='retrain' if args.train else 'saved-weight inference',
                  n_points=len(d), n_dates=len(dates), n_input_features=len(feature_names),
                  feature_columns=feature_names, ensemble_members=len(model.nets),
                  hidden_units=lock['hidden'], final_updates=lock['steps'],
                  shrink_alpha=lock['shrink_alpha'], checks=checks,
                  python=platform.python_version(), torch=torch.__version__,
                  seconds=time.perf_counter()-started)
    (out/'reproduction_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not all(c['matches_archived_metrics'] for c in checks):
        raise SystemExit('Metric mismatch: see generated/reproduction_check.json')
    print('All three archived main-model metrics reproduced.',flush=True)


if __name__ == '__main__':
    main()
