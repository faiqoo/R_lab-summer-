#!/usr/bin/env bash
set -e

# This script is meant to only be called via the Makefile not independently.

function generate_jwt_keys() {
	openssl genrsa -out /tmp/private.key 2048 &>/dev/null
	openssl rsa -pubout -in /tmp/private.key -out /tmp/public.key &>/dev/null
}

function random_secret() {
	local characters=${1}
	local size=${2}
	local name=${3}
	tr -dc "${characters}" </dev/urandom | head -c "${size}" >/secrets/live/"${name}"
}

function main() {
	echo "Generating Secrets"
	local secret_templates=($(find ../secrets/template/* -exec basename {} \;))
	generate_jwt_keys
	for secret in "${secret_templates[@]}"; do
		case "${secret}" in
		DRUPAL_DEFAULT_CONFIGDIR)
			cp /secrets/template/DRUPAL_DEFAULT_CONFIGDIR /secrets/live/DRUPAL_DEFAULT_CONFIGDIR
			;;
		DRUPAL_DEFAULT_SALT)
			random_secret 'A-Za-z0-9-_' 74 DRUPAL_DEFAULT_SALT
			;;
		JWT_ADMIN_TOKEN)
			random_secret 'A-Za-z0-9' 64 JWT_ADMIN_TOKEN
			;;
		JWT_PRIVATE_KEY)
			cp /tmp/private.key /secrets/live/JWT_PRIVATE_KEY
			;;
		JWT_PUBLIC_KEY)
			cp /tmp/public.key /secrets/live/JWT_PUBLIC_KEY
			;;
		*)
			random_secret 'A-Za-z0-9' 48 "${secret}"
			;;
		esac
		echo "Generated or copied ${secret}!"
	done

	# Make sure they are only readable by their owner.
	chmod 600 /secrets/live/*
}
main

# This was added as a custom function to create the JWKS for envoy
# Converts a PEM public key file to JWKS format using predefined variables

set -e

# --- Input/Output files ---
pubkey="../secrets/live/JWT_PUBLIC_KEY"   # Path to your PEM file
jwks="../secrets/live/JWT_JWKS"           # Path for the output JWKS file

# Check if PEM file exists
if [ ! -f "$pubkey" ]; then
  echo "Error: PEM file '$pubkey' not found."
  exit 1
fi

# Extract modulus (n) and exponent (e) from PEM
MOD_HEX=$(openssl rsa -in "$pubkey" -pubin -modulus -noout | cut -d'=' -f2)
EXP_HEX=$(openssl rsa -in "$pubkey" -pubin -text -noout | grep "Exponent:" | awk '{print $2}')

# Function to convert hex to Base64 URL safe
hex_to_base64url() {
  local hex="$1"
  echo -n "$hex" | xxd -r -p | openssl base64 -A | tr '+/' '-_' | tr -d '='
}

# Function to convert decimal number to Base64 URL safe
dec_to_base64url() {
  local dec="$1"
  local hex
  hex=$(printf '%x' "$dec")
  # Ensure even number of hex digits
  (( ${#hex} % 2 == 1 )) && hex="0$hex"
  echo -n "$hex" | xxd -r -p | openssl base64 -A | tr '+/' '-_' | tr -d '='
}

# Convert modulus and exponent
N=$(hex_to_base64url "$MOD_HEX")
E=$(dec_to_base64url "$EXP_HEX")

# Create JWKS JSON
cat > "$jwks" <<EOF
{
  "keys": [
    {
      "kty": "RSA",
      "use": "sig",
      "alg": "RS256",
      "n": "$N",
      "e": "$E"
    }
  ]
}
EOF

echo "JWKS file created at $jwks"

