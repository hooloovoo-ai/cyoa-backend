#!/bin/bash

python3 -m gunicorn -b "0.0.0.0:443" --certfile ../fullchain.pem --keyfile ../privkey.pem server:app --timeout 600