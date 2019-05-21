NAME=$1
cargo web start --target=wasm32-unknown-unknown --auto-reload --example $NAME
