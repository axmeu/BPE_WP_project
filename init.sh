#!/bin/bash
# init.sh
curl -fsSL https://pixi.sh/install.sh | bash
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.profile
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bash_profile
