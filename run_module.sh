relations=(
 "people@person@birth_of_place"
 "people@person@nationality"
 "film@film@language"
)
#relations=("@tv@tv_program@languages")

for relation in ${relations[*]}; do
    python3 filter_train.py -p $i -r $relation -t $type_en  -i 200 -d fb15k
done