#!/usr/bin/env bash

source "$(dirname $0)/_docker_utils.sh"
get_sudo
# --- Original command (commented out for reference) ---
# sudo docker build --build-arg HOST_UID=$(id -u) -t ithemal:latest "$(dirname $0)"

# --- Modified command using buildx to target linux/amd64 ---
# Explanation of flags:
# --platform linux/amd64 : Explicitly tells buildx to build an image for the x86_64 architecture using QEMU emulation.
# --build-arg HOST_UID=$(id -u) : Passes the host user ID (same as before).
# -t ithemal:latest : Tags the image (same as before).
# --load : Loads the resulting multi-platform image into the local Docker image store with the specified tag.
# "$(dirname $0)" : Specifies the build context directory (same as before).

echo "Attempting build for linux/amd64 platform using buildx..."
sudo docker buildx build --platform linux/amd64 --build-arg HOST_UID=$(id -u) -t ithemal:latest --load "$(dirname $0)"

# Check the exit code of the buildx command
exit_code=$?
if [ $exit_code -eq 0 ]; then
  echo "Buildx build completed successfully."
else
  echo "Buildx build failed with exit code $exit_code."
fi

exit $exit_code
