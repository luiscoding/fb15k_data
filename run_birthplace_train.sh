#!/usr/bin/env bash
relations=(
 "people@person@place_of_birth"
 # "people@person@nationality"
# "film@film@language"
)
#relations=("@tv@tv_program@languages")

for relation in ${relations[*]}; do
    python3 filter_train.py  $relation &
done