#!/usr/bin/env bash
relations=(
 "people@person@birth_of_place"
 # "people@person@nationality"
 #"film@film@language"
)
#relations=("@tv@tv_program@languages")

for relation in ${relations[*]}; do
    python3 filter_test.py  $relation
done