#!/bin/bash

# Function to update repository
update_repo() {
    echo "Checking for updates..."
    
    # Fetch latest changes without merging
    git fetch origin

    # Get current and remote hash
    LOCAL_HASH=$(git rev-parse HEAD)
    REMOTE_HASH=$(git rev-parse origin/feat/auto-update)

    if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
        echo "Updates available. Updating repository..."
        
        # Stash any local changes
        git stash
        
        # Pull latest changes
        git pull origin main
        
        # Reinstall dependencies
        pip install -e .
        pip install "numpy<2"
        
        # Restart PM2 services if they exist
        if pm2 list | grep -q "condense_validator"; then
            echo "Restarting validator services..."
            pm2 restart condense_validator
            pm2 restart condense_validator_backend
        fi
        
        echo "Update completed successfully!"
    else
        echo "Repository is up to date!"
    fi
}

# Run the update function
while true; do
    update_repo
    sleep 1800  # Sleep for 30 minutes (1800 seconds)
done
