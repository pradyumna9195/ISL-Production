from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parent
MODEL_H5 = ROOT.parent / 'action_best.h5'
TFJS_DIR = ROOT / 'static' / 'models' / 'action_best_tfjs'


def main() -> int:
    if not MODEL_H5.exists():
        print(f'Model file not found: {MODEL_H5}')
        return 1

    TFJS_DIR.mkdir(parents=True, exist_ok=True)

    print('Converting Keras .h5 model to TensorFlow.js format...')
    converter_executable = ROOT / '.venv' / 'bin' / 'tensorflowjs_converter'
    if not converter_executable.exists():
        fallback = shutil.which('tensorflowjs_converter')
        if not fallback:
            print('Could not find tensorflowjs_converter executable.')
            return 1
        converter_executable = Path(fallback)
    cmd = [
        str(converter_executable),
        '--input_format=keras',
        '--output_format=tfjs_layers_model',
        str(MODEL_H5),
        str(TFJS_DIR / 'model.json'),
    ]

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        print('Conversion failed:')
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode

    print('Conversion succeeded.')
    print(f'TFJS model saved to: {TFJS_DIR / "model.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
