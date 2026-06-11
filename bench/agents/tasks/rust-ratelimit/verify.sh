#!/bin/sh
# jail-friendly: no $HOME assumption, system cargo
CARGO_HOME="$PWD/.cargo_home" exec /usr/bin/cargo test
