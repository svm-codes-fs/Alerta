"""Start the Alerta web dashboard:  python run_dashboard.py   (then open http://127.0.0.1:5000)"""
import os
import sys

# use the locally installed OpenCV 4 / pygame / Flask in .deps if present
DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.deps')
if os.path.isdir(DEPS):
    sys.path.insert(0, DEPS)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from alerta.server import main  # noqa: E402

if __name__ == '__main__':
    main()
