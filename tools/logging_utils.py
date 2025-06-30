import logging
import datetime
import json
import os

class ExtraFormatter(logging.Formatter):
    def format(self, record):
        base = super().format(record)
        # Always show extras that are not standard LogRecord attributes
        standard = logging.LogRecord('', '', '', '', '', '', '', '').__dict__
        extras = {k: v for k, v in record.__dict__.items() if k not in standard and k != 'msg' and k != 'args'}
        if extras:
            base += " | extras: " + json.dumps(extras, default=str)
        return base 