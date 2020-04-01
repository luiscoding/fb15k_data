import sys
from ast import literal_eval
import jsonlines
f = "rules/test_train.jsonl"
rules ={}
with jsonlines.open(f) as reader:
    for obj in reader:
        for k in obj:
            rules[k] = obj[k]["rules"].keys()
for k in rules:
    rule = []
    for ii in rules[k]:
        rule.append(literal_eval(ii.strip("()")))
    rules[k] = rule

relations = [
    "sports@sports_team@sport",
    "people@person@place_of_birth",
     "people@person@nationality",
    "film@film@language",
    "film@director@film",
    "film@director@film",
    "film@film@written_by",
    "tv@tv_program@languages",
   # "location@capital_of_administrative_division@capital_of.@location@administrative_division_capital_relationship@administrative_division" \
    "organization@organization_founder@organizations_founded" ,
     "music@artist@origin"
    ]

for relation in relations:
    rel = "/"+relation.replace("@","/")
    dataPath = "./FB15k-237/"
    rulesPath = dataPath + 'tasks/' + relation + '/' + 'rules.txt'
    with open(rulesPath,"w") as f:
        for ii in rules[rel]:
            if(rel not in ii):
                f.write("\t".join(list(ii)))
                f.write("\n")
                f.flush()