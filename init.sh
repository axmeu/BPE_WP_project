#!/bin/bash
# init.sh

curl -fsSL https://pixi.sh/install.sh | bash
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.profile
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bash_profile

cd ~/work
pixi install
pixi run python -m ipykernel install --user --name pixi-env --display-name "pixi-env"
