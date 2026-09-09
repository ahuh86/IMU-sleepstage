import argparse
from imu_rnn.experiment import Config, run_experiment


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description='CNN + causal vanilla RNN IMU sleep staging')
    for field in ('data_root', 'output_dir', 'cache_dir'):
        parser.add_argument('--'+field.replace('_', '-'), default=getattr(defaults, field))
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--test-subject')
    choice.add_argument('--all-subjects', action='store_true')
    parser.add_argument('--smoke', action='store_true', help='One epoch, whole-night spaced targets per subject')
    parser.add_argument('--resume', action='store_true', help='Reuse verified complete folds; restart an incomplete fold')
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default=defaults.device)
    for field in ('epochs','batch_size','seq_len','patience','seed','num_workers','threads'):
        parser.add_argument('--'+field.replace('_','-'), type=int, default=getattr(defaults, field))
    parser.add_argument('--lr', type=float, default=defaults.lr)
    summary = run_experiment(Config(**vars(parser.parse_args())))
    print(f'Finished {len(summary["folds"])} folds; pooled macro-F1={summary["pooled"]["macro_f1"]:.5f}')


if __name__ == '__main__':
    main()
