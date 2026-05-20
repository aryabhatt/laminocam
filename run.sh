#! /bin/bash

configs=(
    # Add your config file paths here
    #"emma/665/cg/config0.toml"
    #"emma/665/cg/config45.toml"
    "emma/masked/tv/config0.toml"
    "emma/masked/tv/config45.toml"
)

# Check if all files exist first
for config in "${configs[@]}"; do
    if [ ! -f "$config" ]; then
        echo "File not found: $config"
        exit 1
    fi
done

# Run all configs
for config in "${configs[@]}"; do
    ./release/recon "$config"
done
