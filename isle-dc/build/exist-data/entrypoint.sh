#!/bin/sh
set -e

: "${UPSTREAM_HOST:?UPSTREAM_HOST not set}"
: "${UPSTREAM_PORT:?UPSTREAM_PORT not set}"
: "${TOKEN_ISSUER:?TOKEN_ISSUER not set}"

sed \
  -e "s|__UPSTREAM_HOST__|${UPSTREAM_HOST}|g" \
  -e "s|__UPSTREAM_PORT__|${UPSTREAM_PORT}|g" \
  -e "s|__TOKEN_ISSUER__|${TOKEN_ISSUER}|g" \
  -e "s|__JWT_JWKS_CONTENT__|$(cat /run/secrets/JWT_JWKS | tr -d '\n')|g" \
  /etc/envoy/envoy-override.yaml.tmpl > /etc/envoy/envoy-override.yaml

exec envoy -c /etc/envoy/envoy.yaml --config-yaml "$(cat /etc/envoy/envoy-override.yaml)"
