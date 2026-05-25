#!/bin/bash
# init.sh

curl -fsSL https://pixi.sh/install.sh | bash
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.pixi/bin:$PATH"
