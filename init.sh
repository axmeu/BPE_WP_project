#!/bin/bash
# init.sh

curl -fsSL https://pixi.sh/install.sh | bash
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.pixi/bin:$PATH"

cd ~/work
pixi run python -m ipykernel install --user --name pixi-env --display-name "pixi-env"