"""python -m gauge_reader <cameras|calibrate|run|evaluate> - see gauge_reader/cli.py."""
import sys

from arm1.gauge_reader.cli import main

sys.exit(main())
