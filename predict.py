import argparse
from imu_rnn.inference import run_inference


def main():
    parser = argparse.ArgumentParser(description='Predict from a saved CNN-RNN checkpoint')
    parser.add_argument('--checkpoint', dest='checkpoint_path', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--subjects', nargs='+')
    parser.add_argument('--cache-dir')
    rows = run_inference(**vars(parser.parse_args()).copy())
    print(f'Saved {len(rows)} predictions')


if __name__ == '__main__':
    main()
