#!/bin/bash

# Function to check if the user has root privileges
check_root() {
    if [ "$(id -u)" -ne 0 ]; then
        echo "You are not running as root. Commands requiring root will use 'sudo'."
        SUDO="sudo"
    else
        echo "You are running as root. 'sudo' is not required."
        SUDO=""
    fi
}

# Run the root check
check_root

# Update the package list
$SUDO apt update

# Install Redis
$SUDO apt install -y redis

# Verify installation
if redis-cli --version; then
    echo "Redis installed successfully."
else
    echo "Redis installation failed."
    exit 1
fi

# Enable Redis to start on boot
$SUDO systemctl enable redis

# Start Redis service
$SUDO systemctl start redis

# Test Redis
if redis-cli ping | grep -q "PONG"; then
    $SUDO systemctl status redis
    echo "Redis is working correctly!"
else
    echo "Redis test failed. Check the service status with '$SUDO systemctl status redis'."
fi
